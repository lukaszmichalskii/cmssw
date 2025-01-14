import re
import sys

def extract_time(line):
    """
    Extracts the time in microseconds from the string in the format:
    "Unpacking OK (199 us)" or "W3Pi OK (12 us)"
    Returns the time as an integer.
    """
    match = re.search(r'\((\d+)\s+us\)', line)
    if match:
        return int(match.group(1))
    return None

def calculate_averages(input_file):
    """
    Reads the input file, extracts time measurements, and computes:
    - overall average time
    - average for lines with idx % 2 == 0
    - average for lines with idx % 2 == 1
    """
    with open(input_file, 'r') as fd:
        times = []
        even_times = []
        odd_times = []
        idx = 0
        for line in fd:
            time = extract_time(line)
            if time is not None:
                times.append(time)
                if idx % 2 == 0:
                    even_times.append(time)
                else:
                    odd_times.append(time)
                idx += 1

        # Calculate averages
        overall_avg = sum(times) / len(times) if times else 0
        even_avg = sum(even_times) / len(even_times) if even_times else 0
        odd_avg = sum(odd_times) / len(odd_times) if odd_times else 0

        return overall_avg, even_avg, odd_avg

if __name__ == "__main__":
    # Ensure that the input file path is passed as an argument
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]  # The first command-line argument should be the input file path

    overall_avg, even_avg, odd_avg = calculate_averages(input_file)

    print(f"Real time {overall_avg:.0f} us")
    print(f"Unpacking: {even_avg:.0f} us")
    print(f"Combinatorial: {odd_avg:.0f} us")

