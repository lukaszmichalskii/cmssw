from prometheus_client import start_http_server, Counter, Gauge
import asyncio
import os
import pathlib
import re
import time

class Context:
    def __init__(self, ntasks, metrics):
        self.metrics = metrics
        self.ntasks = ntasks
        self.time = [0. for i in range(ntasks)]
        self.norbits = [0 for i in range(ntasks)]
        self.ntruncorbits = [0 for i in range(ntasks)]
        self.nevents = [0 for i in range(ntasks)]
        self.nprocesses = [0 for i in range(ntasks)]
        self.gauges = dict((k, [0 for i in range(ntasks)]) for k in ("threads","vsize","rss"))
        self.timeunits = 1.0/os.sysconf(os.sysconf_names['SC_CLK_TCK'])
        self.pagesize = os.sysconf(os.sysconf_names['SC_PAGESIZE'])
    def updateForStart(self, itask):
        self.nprocesses[itask] = 1
        self.metrics["processes"].update(sum(self.nprocesses))
    def updateForEnd(self, itask):
        self.nprocesses[itask] = 0
        self.metrics["processes"].update(sum(self.nprocesses))
    def updateResources(self, itask, orbits, truncorbits, events, info = None):
        self.metrics["orbits"].inc(orbits - self.norbits[itask])
        self.norbits[itask] = orbits
        self.metrics["truncorbits"].inc(truncorbits - self.ntruncorbits[itask])
        self.ntruncorbits[itask] = truncorbits
        self.metrics["events"].inc(events - self.nevents[itask])
        self.nevents[itask] = events
        if info:
            self.metrics["time"].inc(info["time"] - self.time[itask])
            self.time[itask] = info["time"]
            for k, g in self.gauges.items():
                g[itask] = info[k]
                self.metrics[k].update(sum(g))

def gatherInfo(procpath, ctx : Context):
    fields = procpath.open().readline().split()
    return dict(
        state = fields[2],
        time = sum(int(f) for f in fields[13:17])*ctx.timeunits,
        threads = int(fields[19]),
        vsize = int(fields[22]),
        rss = int(fields[23])*ctx.pagesize,
    )

async def cmsRun(args, itask, ctx : Context):
    proc = await asyncio.create_subprocess_exec(*(["cmsRun"]+args+[f"task={itask}"]),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT)
    print(f"Spawned cmsRun process with pid {proc.pid}")
    ctx.updateForStart(itask)
    statpath = pathlib.Path(f"/proc/{proc.pid}/stat")
    fullreport = []
    orbits, truncorbits, events = 0, 0, 0
    last = time.monotonic()
    pattern = re.compile(r"Processed +(\d+) +orbits, of which +(\d+) +truncated, and +(\d+) +events.")
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        sline = line.decode('ascii').rstrip()
        if sline.startswith("Begin processing the "):
            pass
        elif sline.startswith("Processed"):
            m = re.match(pattern, sline)
            if m:
                orbits = int(m.group(1))
                truncorbits = int(m.group(2))
                events = int(m.group(3))
                info = None
                if time.monotonic() - last > 1:
                    info = gatherInfo(statpath, ctx)
                    print(f"[task {itask:2d}]: processed {orbits:10d} orbits ({truncorbits:9d} truncated), {events:12d} events", flush=True)
                    last = time.monotonic()
                ctx.updateResources(itask, orbits, truncorbits, events, info)
        else:
            fullreport.append(sline)
    await proc.wait()
    ctx.updateForEnd(itask)
    print(f"Done, exited with {proc.returncode}, processed {orbits} orbits ({truncorbits} truncated), {events} events")
    report = f'report.{os.uname()[1]}.{itask}.log'
    freport = open(report, "w")
    for line in fullreport: freport.write(line+"\n")
    freport.close()
    print(f"full report of {len(fullreport)} lines written to {report}")

async def main(args, context : Context):
    tasks = [ asyncio.create_task(cmsRun(args, i, context)) for i in range(context.ntasks) ]
    print(f"started all {len(tasks)} cmsRuns at {time.strftime('%X')}")
    for t in tasks:
        await t
    print(f"finished all cmsRuns at {time.strftime('%X')}")
    for itask in range(context.ntasks):
        report = f'report.{os.uname()[1]}.{itask}.log'
        lines = [ l.rstrip() for l in open(report) ]
        print(f"==== OUTPUT REPORT OF TASK {itask} ({len(lines)} lines) ====") 
        if len(lines) > 100:
            lines = lines[:20] + ["[....]"] + lines[-80:]
        for l in lines:
            print(l)
        print("", flush=True)
class myGauge:
    def __init__(self, name, help):
        self._gauge = Gauge(name, help)
    def update(self, value):
        self._gauge.set(value)

class myCounter:
    def __init__(self, name, help):
        self._counter = Counter(name+"_total", help)
        self._value = 0
    def inc(self, value):
        self._counter.inc(value)
    def update(self, value):
        if value > self._value: # might decrease if some processes drop out
            self._counter.inc(value - self._value)
        self._value = value

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser(usage="%prog")
    parser.add_option("-a", dest="address", type=str, default="127.0.0.1", help="Address to bind to (127.0.0.1)")
    parser.add_option("-p", dest="port", type=int, default=8084, help="Port to expose metrics on (8084)")
    parser.add_option("-j", dest="jobs", type=int, default=1, help="Number of cmsRun jobs to start")
    (options,args) = parser.parse_args()
    metrics = dict(
        orbits = myCounter("processing_orbits", "Orbits processed"),
        truncorbits = myCounter("processing_truncated_orbits", "Orbits lost"),
        events = myCounter("processing_events", "Events processed"),
        time = myCounter("processing_cpu_seconds", "CPU seconds of processing"),
        processes = myGauge("processing_nprocesses", "Number of processes"),
        threads = myGauge("processing_nthreads", "Number of threads"),
        vsize = myGauge("processing_vsize_bytes", "Virtual memory size"),
        rss = myGauge("processing_rss_bytes", "RSS memory size"),
    )
    ctx = Context(options.jobs, metrics)
    print(f"Exporting process metrics on http://{options.address}:{options.port}/metrics\n", flush=True)
    start_http_server(options.port, addr=options.address)
    asyncio.run(main(args, ctx))
