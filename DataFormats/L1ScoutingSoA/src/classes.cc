#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/Portable/interface/PortableHostObjectReadRules.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiSoA.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiStruct.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<PuppiSoA>);
SET_PORTABLEHOSTOBJECT_READ_RULES(PortableHostObject<PuppiStruct>);
