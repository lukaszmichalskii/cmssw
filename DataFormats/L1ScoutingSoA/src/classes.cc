#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/Portable/interface/PortableHostObjectReadRules.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiHostCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiHost.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(PuppiHostCollection);
SET_PORTABLEHOSTOBJECT_READ_RULES(PuppiHost);
