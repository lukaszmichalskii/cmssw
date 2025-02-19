#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiHostCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/ClustersHostCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/JetsHostCollection.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(PuppiHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(ClustersHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(JetsHostCollection);
