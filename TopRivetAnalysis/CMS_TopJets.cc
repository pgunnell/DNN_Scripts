// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Tools/BinnedHistogram.hh"
#include "Rivet/Projections/HeavyHadrons.hh"
#include "fastjet/contrib/SoftDrop.hh"

using namespace fastjet;

namespace Rivet {

  // This analysis is a derived from the class Analysis:
  class CMS_TopJets : public Analysis {

 
  private:
    BinnedHistogram<double> _hist_sigma;
    BinnedHistogram<double> _hist_ifleadingsigma;
    BinnedHistogram<double> _hist_leadingsigma;
    Histo1DPtr hadronpT;
    Histo1DPtr hadronpTfraction;
    Histo1DPtr _h_tau1,_h_tau2,_h_tau3,_h_jet_mass,_h_jet_mass_1,_h_jet_mass_2,_h_jet_mass_12,_h_pt_firstjet,_h_pt_secondjet,_h_eta_firstjet,_h_eta_secondjet, _h_Delta_R_subj ,_h_Symz ,_h_MD,_h_DeltaR12, _h_tau12, _h_tau23,hadronpTsubjet1,hadronpTsubjet2,_h_subjet1tagged,_h_subjet2tagged,hadronpTfractionsubjet1,hadronpTfractionsubjet2;

  public:
    // @name Constructors, init, analyze, finalize
    // @{

    // Constructor
    CMS_TopJets()
      : Analysis("CMS_TopJets") {
    }

    // Returns constituents to make it easier to do the filtering
    PseudoJets splitjet(fastjet::PseudoJet jet, double& last_R, const FastJets& fj, bool& unclustered) const {

      // Build a new cluster sequence just using the constituents of this jet.
      fastjet::ClusterSequence cs(fj.clusterSeq()->constituents(jet), fastjet::JetDefinition(fastjet::antikt_algorithm, M_PI/2.));

      // Get the jet back again
      vector<fastjet::PseudoJet> remadeJets = cs.inclusive_jets(0.);

      if ( remadeJets.size() != 1 ) return remadeJets;

      fastjet::PseudoJet remadeJet = remadeJets[0];
      fastjet::PseudoJet parent1, parent2;
      unclustered = false;

      while ( cs.has_parents(remadeJet, parent1, parent2) ) {
        if (parent1.squared_distance(parent2) < 0.09) break;
        if (parent1.m2() < parent2.m2()) {
          fastjet::PseudoJet tmp;
          tmp = parent1; parent1 = parent2; parent2 = tmp;
        }

        double ktdist = parent1.kt_distance(parent2);
        double rtycut2 = 0.3*0.3;
        if (parent1.m() < 0.67*remadeJet.m() && ktdist > rtycut2*remadeJet.m2()) {
          unclustered = true;
          break;
        } else {
          remadeJet = parent1;
        }
      }

      last_R = 0.5 * sqrt(parent1.squared_distance(parent2));
      return cs.constituents(remadeJet);
    }


    fastjet::PseudoJet filterjet(PseudoJets jets, double& stingy_R, const double def_R) const {
      if (stingy_R == 0.0) stingy_R = def_R;
      stingy_R = def_R < stingy_R ? def_R : stingy_R;
      fastjet::JetDefinition stingy_jet_def(fastjet::antikt_algorithm, stingy_R);
      fastjet::ClusterSequence scs(jets, stingy_jet_def);
      vector<fastjet::PseudoJet> stingy_jets = sorted_by_pt(scs.inclusive_jets(0));
      fastjet::PseudoJet reconst_jet(0, 0, 0, 0);
      for (size_t isj = 0; isj < std::min((size_t) 3, stingy_jets.size()); ++isj) {
        reconst_jet += stingy_jets[isj];
      }
      return reconst_jet;
    }


    // These are custom functions for n-subjettiness.
    PseudoJets jetGetAxes(int n_jets, const PseudoJets& inputJets, double subR) const {
      // Sanity check
      if (inputJets.size() < (size_t) n_jets) {
        MSG_ERROR("Not enough input particles.");
        return inputJets;
      }

      // Get subjets, return
      fastjet::ClusterSequence sub_clust_seq(inputJets, fastjet::JetDefinition(fastjet::kt_algorithm, subR, fastjet::E_scheme, fastjet::Best));
      return sub_clust_seq.exclusive_jets(n_jets);
    }


    double jetTauValue(double beta, double jet_rad, const PseudoJets& particles, const PseudoJets& axes, double Rcut) const {
      double tauNum = 0.0;
      double tauDen = 0.0;

      if (particles.size() == 0) return 0.0;

      for (size_t i = 0; i < particles.size(); i++) {
        // find minimum distance (set R large to begin)
        double minR = 10000.0;
        for (size_t j = 0; j < axes.size(); j++) {
          double tempR = sqrt(particles[i].squared_distance(axes[j]));
          if (tempR < minR) minR = tempR;
        }
        if (minR > Rcut) minR = Rcut;
        // calculate nominator and denominator
        tauNum += particles[i].perp() * pow(minR,beta);
        tauDen += particles[i].perp() * pow(jet_rad,beta);
      }

      // return N-subjettiness (or 0 if denominator is 0)
      return safediv(tauNum, tauDen, 0);
    }


    void jetUpdateAxes(double beta, const PseudoJets& particles, PseudoJets& axes) const {
      vector<int> belongsto;
      for (size_t i = 0; i < particles.size(); i++) {
        // find minimum distance axis
        int assign = 0;
        double minR = 10000.0;
        for (size_t j = 0; j < axes.size(); j++) {
          double tempR = sqrt(particles[i].squared_distance(axes[j]));
          if (tempR < minR) {
            minR = tempR;
            assign = j;
          }
        }
        belongsto.push_back(assign);
      }

      // iterative step
      double deltaR2, distphi;
      vector<double> ynom, phinom, den;

      ynom.resize(axes.size());
      phinom.resize(axes.size());
      den.resize(axes.size());

      for (size_t i = 0; i < particles.size(); i++) {
        distphi = particles[i].phi() - axes[belongsto[i]].phi();
        deltaR2 = particles[i].squared_distance(axes[belongsto[i]]);
        if (deltaR2 == 0.) continue;
        if (abs(distphi) <= M_PI) phinom.at(belongsto[i]) += particles[i].perp() * particles[i].phi() * pow(deltaR2, (beta-2)/2);
        else if ( distphi > M_PI) phinom.at(belongsto[i]) += particles[i].perp() * (-2 * M_PI + particles[i].phi()) * pow(deltaR2, (beta-2)/2);
        else if ( distphi < M_PI) phinom.at(belongsto[i]) += particles[i].perp() * (+2 * M_PI + particles[i].phi()) * pow(deltaR2, (beta-2)/2);
        ynom.at(belongsto[i]) += particles[i].perp() * particles[i].rap() * pow(deltaR2, (beta-2)/2);
        den.at(belongsto[i])  += particles[i].perp() * pow(deltaR2, (beta-2)/2);
      }

      // reset to new axes
      for (size_t j = 0; j < axes.size(); j++) {
        if (den[j] == 0.) axes.at(j) = axes[j];
        else {
          double phi_new = fmod( 2*M_PI + (phinom[j] / den[j]), 2*M_PI );
          double pt_new  = axes[j].perp();
          double y_new   = ynom[j] / den[j];
          double px = pt_new * cos(phi_new);
          double py = pt_new * sin(phi_new);
          double pz = pt_new * sinh(y_new);
          axes.at(j).reset(px, py, pz, axes[j].perp()/2);
        }
      }
    }

    // Book histograms and initialize projections:
    void init() {
      
      const FinalState fs;

      // Initialize the projectors:
      addProjection(FastJets(fs, FastJets::ANTIKT, 0.8),"Jets");

      addProjection(HeavyHadrons(Cuts::abseta < 2.5 && Cuts::pT > 10*GeV), "BHadrons");

      int Ptbins=72;
      double Ptbinning[73] = {18, 21, 24, 28, 32, 37, 43, 49, 56, 64, 74, 84,97, 114, 133, 153, 174, 196, 220, 245, 272, 300, 330, 362, 395, 430, 468,507, 548, 592, 638, 686, 737, 790, 846, 905, 967,1032, 1101, 1172, 1248, 1327, 1410, 1497, 1588, 1684, 1784, 1890, 2000,2116, 2238, 2366, 2500, 2640, 2787, 2941, 3103, 3273, 3450, 3637, 3832,4037, 4252, 4477, 4713, 4961, 5220, 5492, 5777, 6076, 6389, 6717, 7000};

      std::vector<double> PtHistobinning;

      for(int i=0;i<Ptbins+1;i++){
	PtHistobinning.push_back(Ptbinning[i]);
      }

      // Book histograms:
      /*_hist_sigma.addHistogram(0.0, 0.5, bookHisto1D("d01-x01-y01",PtHistobinning));
      _hist_sigma.addHistogram(0.5, 1.0, bookHisto1D("d02-x01-y01",PtHistobinning));
      _hist_sigma.addHistogram(1.0, 1.5, bookHisto1D("d03-x01-y01",PtHistobinning));
      _hist_sigma.addHistogram(1.5, 2.0, bookHisto1D("d04-x01-y01",PtHistobinning));
      _hist_sigma.addHistogram(2.0, 2.4, bookHisto1D("d05-x01-y01",PtHistobinning));

      _hist_leadingsigma.addHistogram(0.0, 0.5, bookHisto1D("d01-x01-y02",PtHistobinning));
      _hist_leadingsigma.addHistogram(0.5, 1.0, bookHisto1D("d02-x01-y02",PtHistobinning));
      _hist_leadingsigma.addHistogram(1.0, 1.5, bookHisto1D("d03-x01-y02",PtHistobinning));
      _hist_leadingsigma.addHistogram(1.5, 2.0, bookHisto1D("d04-x01-y02",PtHistobinning));
      _hist_leadingsigma.addHistogram(2.0, 2.4, bookHisto1D("d05-x01-y02",PtHistobinning));

      _hist_ifleadingsigma.addHistogram(0.0, 0.5, bookHisto1D("d01-x01-y03",PtHistobinning));
      _hist_ifleadingsigma.addHistogram(0.5, 1.0, bookHisto1D("d02-x01-y03",PtHistobinning));
      _hist_ifleadingsigma.addHistogram(1.0, 1.5, bookHisto1D("d03-x01-y03",PtHistobinning));
      _hist_ifleadingsigma.addHistogram(1.5, 2.0, bookHisto1D("d04-x01-y03",PtHistobinning));
      _hist_ifleadingsigma.addHistogram(2.0, 2.4, bookHisto1D("d05-x01-y03",PtHistobinning));*/
 
      _h_tau1 = bookHisto1D("tau1", 20, 0.,1.0);
      _h_tau2 = bookHisto1D("tau2", 20, 0.,1.0);  
      _h_tau3 = bookHisto1D("tau3", 20, 0.,1.0);

      _h_tau12 = bookHisto1D("tau12", 20, 0.,1.0);
      _h_tau23 = bookHisto1D("tau23", 20, 0.,1.0);  
      
      _h_jet_mass = bookHisto1D("jet_mass", 200, 0.5,1001.5);     
      _h_jet_mass_1 = bookHisto1D("jet_mass_1", 200, 0.5,1001.5);     
      _h_jet_mass_2 = bookHisto1D("jet_mass_2", 200, 0.5,1001.5);     
      _h_jet_mass_12 = bookHisto1D("jet_mass_12", 200, 0.5,1001.5);     
      
      _h_pt_firstjet = bookHisto1D("pt_firstjet",PtHistobinning);   
      _h_pt_secondjet = bookHisto1D("pt_secondjet",PtHistobinning);  
      
      _h_eta_firstjet = bookHisto1D("eta_firstjet", 20, -5,5);    
      _h_eta_secondjet = bookHisto1D("eta_secondjet", 20, -5,5);     

      _h_DeltaR12 = bookHisto1D("DeltaR12", 20, 0.,1.0);
      
      _h_Delta_R_subj = bookHisto1D("DeltaR_subjet", 20, 0.,1.0);
      _h_Symz = bookHisto1D("Symetric_factor_z", 20, 0.,1.0);
      _h_MD = bookHisto1D("SoftDropMass", 20, 0.,1.0);

      hadronpT = bookHisto1D("hadronpT", 200, 0.5,1001.5);
      hadronpTfraction = bookHisto1D("hadronpTfraction", 20, 0.,1.0);

      hadronpTsubjet1 = bookHisto1D("hadronpTsubjet1", 200, 0.5,1001.5);
      hadronpTsubjet2 = bookHisto1D("hadronpTsubjet2", 200, 0.5,1001.5);
      hadronpTfractionsubjet2 = bookHisto1D("hadronpTfractionsubjet2", 20, 0.,1.0);
      hadronpTfractionsubjet1 = bookHisto1D("hadronpTfractionsubjet1", 20, 0.,1.0);

      _h_subjet1tagged = bookHisto1D("subjet1tagged", 3, -0.5,2.5);
      _h_subjet2tagged = bookHisto1D("subjet2tagged", 3, -0.5,2.5);

    }

    // Analysis
    void analyze(const Event &event) {

      const double weight = event.weight();      
      const FastJets &fj = applyProjection<FastJets>(event,"Jets");      
      const Jets& jets = fj.jetsByPt(Cuts::ptIn(500*GeV, 7000.0*GeV) && Cuts::absrap < 2.2);

      const Particles& bHadrons = applyProjection<HeavyHadrons>(event, "BHadrons").bHadrons();

      int leadingentry=1;
      int btagleadingentry=1;

      double beta = 0.8;
      double Rcut=10.;

      double z_cut = 0.10;
      double beta_SD  = 0.;
      contrib::SoftDrop sd(beta_SD, z_cut,0.8);
      //cout << "SoftDrop groomer is: " << sd.description() << endl;

      foreach (const Jet& j, jets) {
	
	bool btag=false;
	bool btagleading=false;
	
	double leadinghadronpt=-10;

	foreach (const Particle& b, bHadrons){
	  if (deltaR(j, b) < 0.4) { 
	    btag = true; 
	    if(b.momentum().perp()>leadinghadronpt) leadinghadronpt=b.momentum().perp();
	  }
	}
	
	if(btagleadingentry){
	  btagleadingentry=0;
	  foreach (const Particle& b, bHadrons){
	    if (deltaR(j, b) < 0.4) { btagleading = true; break; }
	  }	  
	}

	
	if (btag){ 

	  double dR = 0;
	  bool unclustered = false;
	  PseudoJets split_jets = splitjet(j, dR, fj, unclustered);
	  if ( (dR < 0.15) || (unclustered == false) ) continue;
	  PseudoJet filt_jet = filterjet(split_jets, dR, 0.3);

	  const PseudoJets constituents = fj.clusterSeq()->constituents(j);
	  if (constituents.size() < 3) continue;
	  
	  const PseudoJets axis1 = jetGetAxes(1, constituents, M_PI/2.0);
	  const PseudoJets axis2 = jetGetAxes(2, constituents, M_PI/2.0);
	  const PseudoJets axis3 = jetGetAxes(3, constituents, M_PI/2.0);
	  
	  const double radius = 0.8;
	  const double tau1 = jetTauValue(beta, radius, constituents, axis1, Rcut);
	  const double tau2 = jetTauValue(beta, radius, constituents, axis2, Rcut);
	  const double tau3 = jetTauValue(beta, radius, constituents, axis3, Rcut);

          _h_tau1->fill(tau1,weight);  
          _h_tau2->fill(tau2,weight);  
          _h_tau3->fill(tau3,weight);  

          _h_tau12->fill(tau2/tau1,weight);  
          _h_tau23->fill(tau3/tau2,weight);  

	  //pT of the first three jet constituents
	  
	  PseudoJet sd_jet = sd(j);
	  
	  assert(sd_jet != 0); //because soft drop is a groomer (not a tagger), it should always return a soft-dropped jet
	  
	  //cout << "  delta_R between subjets: " << sd_jet.structure_of<contrib::SoftDrop>().delta_R() << endl;
	  //cout << "  symmetry measure(z):     " << sd_jet.structure_of<contrib::SoftDrop>().symmetry() << endl;
	  //cout << "  mass drop(mu):           " << sd_jet.structure_of<contrib::SoftDrop>().mu() << endl;
 
          _h_Delta_R_subj->fill(sd_jet.structure_of<contrib::SoftDrop>().delta_R(),weight);  
          _h_Symz->fill(sd_jet.structure_of<contrib::SoftDrop>().symmetry(),weight); 
          _h_MD->fill(sd_jet.structure_of<contrib::SoftDrop>().mu(),weight); 
	  
	  vector<PseudoJet> pieces = sd_jet.pieces();
      
	  const double ptfirstconst = pieces[0].perp();
	  const double ptsecondconst = pieces[1].perp();

          const double etafirstconst = pieces[0].eta();
	  const double etasecondconst = pieces[1].eta();
  
          _h_pt_firstjet->fill(ptfirstconst,weight);  
          _h_pt_secondjet->fill(ptsecondconst,weight);  

          _h_eta_firstjet->fill(etafirstconst,weight);  
          _h_eta_secondjet->fill(etasecondconst,weight);  
	  
	  //deltaR of the subjets

	  const double deltaR_12=sqrt(pow(pieces[0].eta()-pieces[1].eta(),2)+pow(deltaPhi(pieces[0].phi(),pieces[1].phi()),2));

	  _h_DeltaR12->fill(deltaR_12,weight);

	  //mass of the jet and of the three subjets
	  const double jet_mass=j.momentum().mass();
	  const double jet_mass_1=pieces[0].m();
	  const double jet_mass_2=pieces[1].m();

         _h_jet_mass->fill(jet_mass,weight);  
         _h_jet_mass_1->fill(jet_mass_1,weight);  
         _h_jet_mass_2->fill(jet_mass_2,weight); 

	  //subjet tagged?
	  bool subjettagged1=false;
	  double pthadronsubjet1=-10;
	  foreach (const Particle& b, bHadrons){
	    if (sqrt(pow(pieces[0].eta()-b.momentum().eta(),2)+pow(deltaPhi(pieces[0].phi(),b.momentum().phi()),2)) < 0.1) { 
	      subjettagged1 = true; 
	      if(pthadronsubjet1<b.momentum().perp()) pthadronsubjet1=b.momentum().perp();
	    }
	  }

	  bool subjettagged2=false;
	  double pthadronsubjet2=-10;
	  foreach (const Particle& b, bHadrons){
	    if (sqrt(pow(pieces[1].eta()-b.momentum().eta(),2)+pow(deltaPhi(pieces[1].phi(),b.momentum().phi()),2)) < 0.1) { 
	      subjettagged2 = true; 
	      if(pthadronsubjet2<b.momentum().perp()) pthadronsubjet2=b.momentum().perp();
	    }
	  }

	  _h_subjet1tagged->fill(subjettagged1,weight);
	  _h_subjet2tagged->fill(subjettagged2,weight);

	  hadronpTsubjet1->fill(pthadronsubjet1,weight);
	  hadronpTsubjet2->fill(pthadronsubjet2,weight);

	  hadronpTfractionsubjet1->fill(pthadronsubjet1/pieces[0].perp(),weight);
	  hadronpTfractionsubjet2->fill(pthadronsubjet2/pieces[1].perp(),weight);
	  
	  const double jet_mass_12=sqrt(pow(pieces[0].E()+pieces[1].E(),2)-pow(pieces[0].px()+pieces[1].px(),2)-pow(pieces[0].py()+pieces[1].py(),2)-pow(pieces[0].pz()+pieces[1].pz(),2));

         _h_jet_mass_12->fill(jet_mass_12,weight);  
         

	  hadronpT->fill(leadinghadronpt,weight); 
	  hadronpTfraction->fill(leadinghadronpt/j.momentum().pT(),weight); 
	  
	  //_hist_sigma.fill(fabs(j.momentum().rapidity()), j.momentum().pT(), weight);
	
	  //OUTPUT FOR MACHINE LEARNING TRAINING
	  cout<<" QCD:-1 TOP:1 "<<subjettagged1<<" "<<subjettagged2<<" "<<pthadronsubjet1<<" "<<pthadronsubjet2<<" "<<pthadronsubjet1/pieces[0].perp()<<" "<<pthadronsubjet2/pieces[1].perp()<<" "<<jet_mass_12<<" "<<leadinghadronpt<<" "<<leadinghadronpt/j.momentum().pT()<<" "<<sd_jet.structure_of<contrib::SoftDrop>().delta_R()<<" "<<sd_jet.structure_of<contrib::SoftDrop>().symmetry()<<" "<<sd_jet.structure_of<contrib::SoftDrop>().mu()<<" "<<ptfirstconst<<" "<<ptsecondconst<<" "<<deltaR_12<<" "<<jet_mass<<" "<<jet_mass_1<<" "<<jet_mass_2<<" "<<tau1<<" "<<tau2<<" "<<tau3<<" "<<tau2/tau1<<" "<<tau3/tau2<<endl;

	
	  if(leadingentry) {
	    //_hist_leadingsigma.fill(fabs(j.momentum().rapidity()), j.momentum().pT(), weight);
	    leadingentry=0;
	  }
	}

	if (btagleading){
	  //_hist_ifleadingsigma.fill(fabs(j.momentum().rapidity()), j.momentum().pT(), weight);
	}


      }

    }
    
    // Finalize
    void finalize() {
      cout<<"cross Section: "<<crossSection()<<endl;
      //double invlumi = crossSection()/picobarn/sumOfWeights(); //norm to cross section
      normalize(hadronpT);
      normalize(hadronpTfraction);
      //_hist_sigma.scale(crossSection()/sumOfWeights()/2, this);
      //_hist_leadingsigma.scale(crossSection()/sumOfWeights()/2, this);
      //_hist_ifleadingsigma.scale(crossSection()/sumOfWeights()/2, this);
      normalize(_h_tau1);
      normalize(_h_tau2);
      normalize(_h_tau3);
      normalize(hadronpTsubjet1);
      normalize(hadronpTfractionsubjet1);
      normalize(hadronpTsubjet2);
      normalize(hadronpTfractionsubjet2);
      normalize(_h_jet_mass);
      normalize(_h_jet_mass_1);
      normalize(_h_jet_mass_2);
      normalize(_h_jet_mass_12);
      normalize(_h_pt_firstjet);
      normalize(_h_pt_secondjet);
      normalize(_h_eta_firstjet);
      normalize(_h_eta_secondjet);
      normalize(_h_Delta_R_subj);
      normalize(_h_Symz);
      normalize(_h_MD);
      normalize(_h_DeltaR12);
      normalize(_h_tau12); 
      normalize(_h_tau23);

    }
    //@}


  };

  // This global object acts as a hook for the plugin system. 
  AnalysisBuilder<CMS_TopJets> plugin_CMS_TopJets;

}
