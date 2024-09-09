#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <vector>
#include <set>

/*
 * Filter orbits that don't contain at least one selected BX
 * from a BxSelector module and produce a vector of selected BXs
 */
class GoodOrbitNBxSelector : public edm::global::EDFilter<> {
public:
  explicit GoodOrbitNBxSelector(const edm::ParameterSet&);
  ~GoodOrbitNBxSelector() {}
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // tokens for BX selected by each analysis
  std::vector<edm::EDGetTokenT<unsigned int>> nbxTokens_;
  unsigned int threshold_;
};

GoodOrbitNBxSelector::GoodOrbitNBxSelector(const edm::ParameterSet& iPSet)
    : threshold_(iPSet.getParameter<unsigned int>("nbxMin")) {
  // get the list of selected BXs
  std::vector<edm::InputTag> bxLabels = iPSet.getParameter<std::vector<edm::InputTag>>("unpackers");
  for (const auto& bxLabel : bxLabels) {
    edm::InputTag tag{bxLabel.label(), "nbx", bxLabel.process()};
    nbxTokens_.push_back(consumes<unsigned int>(tag));
  }
}

// ------------ method called for each ORBIT  ------------
bool GoodOrbitNBxSelector::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  for (const auto& token : nbxTokens_) {
    edm::Handle<unsigned> nbx;
    iEvent.getByToken(token, nbx);
    if (*nbx < threshold_)
      return false;
  }
  return true;
}

void GoodOrbitNBxSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("unpackers");
  desc.add<unsigned int>("nbxMin", 3564);  // BXs in one orbit
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(GoodOrbitNBxSelector);
