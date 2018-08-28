// NRRD Image File Format
#include <NRRD/nrrd_image.hxx>

// Fourier Transform
#include<fftw3.h>

// Simple CUDA Wrappers
#include <LibUtilsCuda/CudaBindlessTexture.h>
typedef UtilsCuda::BindlessTexture2D<float> CudaTexture;

// Utilities for Displaying Images and Plots
#include <LibUtilsQt/Figure.hxx>
#include <LibUtilsQt/Plot.hxx>
using UtilsQt::Figure;
using UtilsQt::Plot;

// A Simple QT Utility to Show a Settings Window
#include <GetSetGui/GetSetGui.h>
GetSetGui::Application g_app("VisualizeECC_RadonIntermediate");

// Computing  Radon transform and evaluating Epipolar Consistency-
#include <LibEpipolarConsistency/Gui/ComputeRadonIntermediate.hxx>
#include <LibEpipolarConsistency/EpipolarConsistencyRadonIntermediate.h>
#include <LibEpipolarConsistency/EpipolarConsistencyRadonIntermediateCPU.hxx>

/// A call-back function to handle GUI-input
void gui(const GetSetInternal::Node& node)
{
	using namespace EpipolarConsistency;

	// When the button has been clicked
	if (node.name == "Update")
	{
		g_app.progressStart("Epipolar Consistency", "Evaluating via Radon intermediate functions...", 7);

		// Load image files
		std::string path0 = GetSet<>("Epipolar Consistency/Images/Image 0");
		std::string path1 = GetSet<>("Epipolar Consistency/Images/Image 1");
		NRRD::Image<float> image0(path0);
		g_app.progressUpdate(1);
		NRRD::Image<float> image1(path1);
		g_app.progressUpdate(2);

		// Get pixel spacing and projection matrices
		using Geometry::ProjectionMatrix;
		double spx = GetSet<double>("Epipolar Consistency/Images/Pixel Spacing");
		ProjectionMatrix P0 = stringTo<ProjectionMatrix>(image0.meta_info["Projection Matrix"]);
		ProjectionMatrix P1 = stringTo<ProjectionMatrix>(image1.meta_info["Projection Matrix"]);

		Filter filter = (Filter)(GetSet<int>("Epipolar Consistency/Radon Intermediate/Distance Filter").getValue());

		// Compute both Radon intermediate functions
		EpipolarConsistency::RadonIntermediateFunction compute_dtr;
		using EpipolarConsistency::RadonIntermediate;
		compute_dtr.gui_retreive_section("Epipolar Consistency/Radon Intermediate");
		g_app.progressUpdate(3);
		RadonIntermediate *rif0 = compute_dtr.compute(image0, &P0, &spx);
		g_app.progressUpdate(4);
		RadonIntermediate *rif1 = compute_dtr.compute(image1, &P1, &spx);
		g_app.progressUpdate(5);

		// Evaluate ECC 
		// (this way of evaluating ECC is usually intended for many more than 2 projections, hence the std::vectors)
		std::vector<ProjectionMatrix> Ps;
		Ps.push_back(P0);
		Ps.push_back(P1);
		std::vector<RadonIntermediate*> rifs;
		rifs.push_back(rif0);
		rifs.push_back(rif1);

		rif0->readback();
		Figure fig0("Radon Intermediate Function 0", rif0->data(), true);
		rif1->readback();
		Figure fig1("Radon Intermediate Function 1", rif1->data(), true);
	
		// Output data
		std::vector<float> redundant_samples0, redundant_samples1;
		std::vector<float> kappas;
		std::vector<std::pair<float,float> > rif_samples0, rif_samples1;
		
		// hier die eigene Funktion mit fourierTransformation und ramp filter implementieren
		using namespace std;
		using namespace EpipolarConsistency;
		int arraysize = GetSet<int>("Epipolar Consistency/Radon Intermediate/Number Of Bins/Angle");
		vector<float> sinc(arraysize);
		vector<float> cosine(arraysize);
		vector<float> ramp(arraysize);
		vector<int> n(arraysize);

		for (int i = 0; i < arraysize; i++) {

				float x = (abs((i - arraysize / 2.0) / (arraysize / 2.0))*Pi / 2.0) - Pi / 2.0;

				n[i] = i - arraysize / 2.0;
				ramp[i] = (arraysize / 2.0 - abs(i - arraysize / 2.0)) / (arraysize / 2.0);
				if (i > arraysize / 2.0) {
					//ramp[i] = 0;
				}
				cosine[i] = cos(x)*ramp[i];
				sinc[i] = ((x != 0) ? sin(x) / (x) : 1);
				sinc[i] *= ramp[i];
		}
		/*
		// hier noch allgemeiner für nicht quadratisch
		int arraysize =  rif0->getRadonBinNumber(0);

		std::vector<NRRD::ImageView<float>> fouriers;
		std::vector<NRRD::ImageView<float>> originals;
		std::vector<NRRD::ImageView<float>> filteredFImages;
		std::vector<NRRD::ImageView<float>> filteredOImages;

		vector<float> sinc(arraysize);
		vector<float> cosine(arraysize);
		vector<float> ramp(arraysize);
		vector<int> n(arraysize);

		NRRD::Image<float> transformed(arraysize, arraysize);
		NRRD::Image<float> inverted(arraysize, arraysize);
		NRRD::Image<float> filteredF(arraysize, arraysize);
		NRRD::Image<float> filteredO(arraysize, arraysize);

		for (int k = 0; k < rifs.size(); k++) {

			rifs[k]->readback();

			for (int j = 0; j < rifs[k]->getRadonBinNumber(1); j++) {

				fftw_plan p, pBack, pBackFiltered;
				vector<fftw_complex> in(arraysize);
				vector<fftw_complex> out(arraysize);
				vector<fftw_complex >infftFiltered(arraysize);
				vector<fftw_complex> outfftFiltered(arraysize);

				p = fftw_plan_dft_1d(arraysize, in.data(), out.data(), FFTW_FORWARD, FFTW_ESTIMATE); //fourier transform
				pBack = fftw_plan_dft_1d(arraysize, out.data(), in.data(), FFTW_BACKWARD, FFTW_ESTIMATE); // backtransform to original image
				pBackFiltered = fftw_plan_dft_1d(arraysize, infftFiltered.data(), outfftFiltered.data(), FFTW_BACKWARD, FFTW_ESTIMATE); //backtransform of filtered image

				for (int i = 0; i < arraysize; i++) {

					
					in[i][0] = rifs[k]->data().pixel(j, i); //real
					in[i][1] = 0.0; //complex

					//init sinc for shepp-logan
					if (k == 0 && j ==0) {
						
						float x = (abs((i - arraysize / 2.0)/(arraysize/2.0))*Pi / 2.0)-Pi/2.0;

						n[i] = i-arraysize/2.0;
						ramp[i] = (arraysize/2.0 -  abs(i - arraysize / 2.0))/(arraysize/2.0);
						if (i > arraysize / 2.0) {
							//ramp[i] = 0;
						}
						cosine[i] = cos(x)*ramp[i];
						sinc[i] = ((x != 0)? sin(x) / (x) : 1);
						sinc[i] *= ramp[i];
					}
				}
				//execute  fourier
				fftw_execute(p);

				for (int i = 0; i < arraysize; i++) {
					
					switch (filter) {

					case Filter::Ramp:
						//ramp
						infftFiltered[i][0] = out[i][0] * ramp[i];
						infftFiltered[i][1] = out[i][1] * ramp[i];
						break;

					case Filter::SheppLogan:
						//shepp-logan
						infftFiltered[i][0] = out[i][0] * sinc[i];
						infftFiltered[i][1] = out[i][1] * sinc[i];
						break;
					case Filter::Cosine:
						//cosine
						infftFiltered[i][0] = out[i][0] * cosine[i];
						infftFiltered[i][1] = out[i][1] * cosine[i];
						break;

					default:
						//no filtering
						infftFiltered[i][0] = out[i][0];
						infftFiltered[i][1] = out[i][1];
					}

					//simply setting an logarithmic scale for visualization
					transformed.pixel(j,i) = (float)log(sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]) + 1.0);
					filteredF.pixel(j,i) = (float)log(sqrt(infftFiltered[i][0] * infftFiltered[i][0] + infftFiltered[i][1] * infftFiltered[i][1]) + 1.0);
				}
				fftw_execute(pBack);
				fftw_execute(pBackFiltered);

				for (int i = 0; i < arraysize; i++) {

					//setting magnitude part to filtered image
					float temp1 = outfftFiltered[i][0] / arraysize;
					float temp2 = outfftFiltered[i][1] / arraysize;

					filteredO.pixel(j,i) = sqrt(temp1*temp1 + temp2*temp2);

					//setting and scaling the backtransformed image without filtering for clarity
					//inverted.pixel(j,i) = sqrt(in[i][0]* in[i][0] + in[i][1] * in[i][1] )/ arraysize;
					inverted.pixel(j, i) = in[i][0]/arraysize;

				}
				fftw_destroy_plan(p);
				fftw_destroy_plan(pBack);
				fftw_destroy_plan(pBackFiltered);
				
			}

			fouriers.push_back(transformed);
			originals.push_back(inverted);
			filteredFImages.push_back(filteredF);
			filteredOImages.push_back(filteredO);

			std::map<std::string, std::string> dict;
			rifs[k]->writePropertiesToMeta(dict);
			//sending computed data to rif[k] at gpu level 
			rifs[k]->replaceRadonIntermediateData(filteredOImages[k]);
			rifs[k]->readPropertiesFromMeta(dict);//wie geht das eleganter?
		}
	*/
		// Slow CPU evaluation for just two views
		// (usually, you just call ecc.evaluate(...) which uses the GPU to compute metric for all projections)
		EpipolarConsistency::MetricRadonIntermediate ecc(Ps,rifs);
		ecc.setdKappa(GetSet<double>    ("Epipolar Consistency/Sampling/Angle Step (deg)")/180*Geometry::Pi);
		double inconsistency=ecc.evaluateForImagePair(0,1, &redundant_samples0, &redundant_samples1,&kappas, &rif_samples0, &rif_samples1);
		cout << "inconsistency: " << inconsistency << endl;

		g_app.progressUpdate(6);

		// Visualize redundant signals
		QColor red(255,0,0);
		QColor black(0,0,0);
		QColor blue(0, 0, 255);
		Plot plot("ECC using Radon intermediate functions");
		plot.closeAll();
		plot.graph()
			.setData((int)kappas.size(),kappas.data(),redundant_samples0.data())
			.setName("Redundant Samples 0")
			.setColor(black);
		plot.graph()
			.setData((int)kappas.size(),kappas.data(),redundant_samples1.data())
			.setName("Redundant Samples 1")
			.setColor(red);
		plot.setAxisAngularX();
		plot.setAxisLabels("Epipolar Plane Angle","Radon Intermediate Values [a.u.]");
		
		Plot filterPlot("used filters in frequency domain");
		filterPlot.graph().setData(arraysize, n.data(), sinc.data()).setName("sinc").setColor("red");
		filterPlot.graph().setData(arraysize, n.data(), cosine.data()).setName("cosine").setColor("black");
		filterPlot.graph().setData(arraysize, n.data(), ramp.data()).setName("ramp").setColor("blue");
		filterPlot.showLegend();

		// Show Radon intermediate functions.

		//Figure fig2("Magnitude columnwise fourier transformed", fouriers[0]);
		//Figure fig3("Backtransformed image without filtering", originals[0]);
		//Figure fig4("Ramp filtered FFT", filteredFImages[0]);
		//Figure fig5("Backtransformed ramp-filtered image ", filteredOImages[0]);
		

#ifndef DEBUG
		/*
		// Show sample locations (slow)
		using namespace std;
		bool check = rif0->isDerivative();
		double n_alpha2=0.5*rif0->data().size(0);
		double n_t2    =0.5*rif0->data().size(1);
		double step_alpha=rif1->getRadonBinSize(0);
		double step_t=rif1->getRadonBinSize(1);
		
		for (int i = 0; i < (int)rif_samples0.size(); i+=8) {
			fig0.drawPoint(rif_samples0[i].first / step_alpha + n_alpha2, rif_samples0[i].second / step_t + n_t2, black);
			for (int i=0; i<(int)rif_samples1.size(); i++)
			cout << rif_samples0[i].second / step_t + n_t2 << endl;
			fig1.drawPoint(rif_samples1[i].first / step_alpha + n_alpha2, rif_samples1[i].second / step_t + n_t2, red);
		}
		*/
#endif

		g_app.progressEnd();
	}
		
	// Allow user to edit and save plots as pdf
	if (node.name=="Show Plot Editor...")
		UtilsQt::showPlotEditor();

	// Write ini-File
	g_app.saveSettings();
}

/// Main: Show little window with settings.
int main(int argc, char ** argv)
{
	// Define default settings
	GetSetGui::File   ("Epipolar Consistency/Images/Image 0"           ).setExtensions("2D NRRD image (*.nrrd);All Files (*)");
	GetSetGui::File   ("Epipolar Consistency/Images/Image 1"           ).setExtensions("2D NRRD image (*.nrrd);All Files (*)");
	GetSet<double>    ("Epipolar Consistency/Images/Pixel Spacing"     )=.308;
	GetSetGui::Section("Epipolar Consistency/Images"                   ).setGrouped();
	GetSet<double>    ("Epipolar Consistency/Sampling/Angle Step (deg)")=0.01;

	EpipolarConsistency::RadonIntermediateFunction().gui_declare_section("Epipolar Consistency/Radon Intermediate");
	GetSetGui::Button("Epipolar Consistency/Update")="Update...";

	// Run application
	g_app.init(argc,argv,gui);
	g_app.window().addMenuItem("File","Show Plot Editor...");
	g_app.window().addMenuItem("File"),
	g_app.window().addDefaultFileMenu();
	g_app.window().aboutText()=
		"<h4>Visualization of Epipolar Consistency and Fan-Beam Consistency.</h4>\n\n"
		"Copyright 2014-2018 by <a href=\"mailto:aaichert@gmail.com?Subject=[Epipolar Consistency]\">Andre Aichert</a> <br>"
		"<h4>Epipolar Consistency:</h4>\n\n"
		"Any two ideal transmission images with perfectly known projection geometry contain redundant information. "
		"Inconsistencies, i.e., motion, truncation, scatter radiation or beam-hardening can be observed using Epipolar Consistency. "
		"<br>"
		"<br>"
		"See also: "
		"<br>"
		"<a href=\"https://www5.cs.fau.de/research/software/epipolar-consistency/\">Pattern Recognition Lab at Technical University of Erlangen-Nuremberg</a> "
		"<br>"
		"<h4>Licensed under the Apache License, Version 2.0 (the \"License\")</h4>\n\n"
		"You may not use this file except in compliance with the License. You may obtain a copy of the License at "
		"<a href=\"http://www.apache.org/licenses/LICENSE-2.0\">http://www.apache.org/licenses/LICENSE-2.0</a><br>"
		;
	return g_app.exec();
}
