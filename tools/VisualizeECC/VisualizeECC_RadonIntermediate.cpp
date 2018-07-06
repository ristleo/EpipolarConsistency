// NRRD Image File Format
#include <NRRD/nrrd_image.hxx>

// Fourier Transform
#include<fftw3.h>
#include<unsupported\Eigen\FFT>

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
	if (node.name=="Update")
	{
		g_app.progressStart("Epipolar Consistency","Evaluating via Radon intermediate functions...",7);

		// Load image files
		std::string path0=GetSet<>("Epipolar Consistency/Images/Image 0");
		std::string path1=GetSet<>("Epipolar Consistency/Images/Image 1");
		NRRD::Image<float> image0(path0);
		g_app.progressUpdate(1);
		NRRD::Image<float> image1(path1);
		g_app.progressUpdate(2);

		// Get pixel spacing and projection matrices
		using Geometry::ProjectionMatrix;
		double spx=GetSet<double>("Epipolar Consistency/Images/Pixel Spacing");
		ProjectionMatrix P0=stringTo<ProjectionMatrix>(image0.meta_info["Projection Matrix"]);
		ProjectionMatrix P1=stringTo<ProjectionMatrix>(image1.meta_info["Projection Matrix"]);
	
		// Compute both Radon intermediate functions
		EpipolarConsistency::RadonIntermediateFunction compute_dtr;
		using EpipolarConsistency::RadonIntermediate;
		compute_dtr.gui_retreive_section("Epipolar Consistency/Radon Intermediate");
		g_app.progressUpdate(3);
		RadonIntermediate *rif0=compute_dtr.compute(image0,&P0,&spx);
		g_app.progressUpdate(4);
		RadonIntermediate *rif1=compute_dtr.compute(image1,&P1,&spx);
		g_app.progressUpdate(5);
		
		// Evaluate ECC 
		// (this way of evaluating ECC is usually intended for many more than 2 projections, hence the std::vectors)
		std::vector<ProjectionMatrix> Ps;
		Ps.push_back(P0);
		Ps.push_back(P1);
		std::vector<RadonIntermediate*> rifs;
		rifs.push_back(rif0);
		rifs.push_back(rif1);
		// Output data
		std::vector<float> redundant_samples0, redundant_samples1;
		std::vector<float> kappas;
		std::vector<std::pair<float,float> > rif_samples0, rif_samples1;
		
		// TODO hier die eigene Funktion mit fourierTransformation und ramp filter implementieren
		//fftwf_execute_dft()
		using namespace std;
		using namespace EpipolarConsistency;
		std::vector<float*> cols0;
		int arraysize =  rif0->getRadonBinNumber(0);
		rif0->readback();




		//initialize size of pointers

		float* transformed = (float*) malloc(sizeof(float)*arraysize*arraysize);
		float* inverted = (float*)malloc(sizeof(float)*arraysize*arraysize);
		float* filteredF = (float*)malloc(sizeof(float)*arraysize*arraysize);
		float* filteredO = (float*)malloc(sizeof(float)*arraysize*arraysize);


		float maxF = 0;
		float maxO = 0;


		for (int j = 0; j < rif0->getRadonBinNumber(1); j++) {
			//float* cur_col = new float[arraysize]();
			fftw_complex *in, *out, *infftFiltered, *outfftFiltered;
			fftw_plan p, pBack, pBackFiltered;

			in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*arraysize);
			out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*arraysize);
			infftFiltered = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*arraysize);
			outfftFiltered = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*arraysize);


			p = fftw_plan_dft_1d(arraysize, in, out, FFTW_FORWARD, FFTW_ESTIMATE); //fourier transform
			pBack = fftw_plan_dft_1d(arraysize, out, in, FFTW_BACKWARD, FFTW_ESTIMATE); // backtransform to original image
			pBackFiltered = fftw_plan_dft_1d(arraysize, infftFiltered, outfftFiltered, FFTW_BACKWARD, FFTW_ESTIMATE); //backtransform of filtered image


			for (int i = 0; i < arraysize; i++) {
			
				in[i][0] = rif0->data().pixel(j, i); //real
				in[i][1] = 0.0; //complex



				//cout <<"in "<< in[i][0] << endl;
				//cout << "data " << rif0->data().pixel(i, j) << endl;

				//cur_col[i] = rif0->data().pixel(i, j);
				//ges[i+arraysize*j] = rif0->data().pixel( j, i);
			}
			//cols0.push_back(cur_col);
			fftw_execute(p);
			
			for (int i = 0; i < arraysize; i++) {

				//simply setting an logarithmic scale for output
				transformed[j + arraysize*i] = (float) log(sqrt(out[i][0]*out[i][0]+ out[i][1] * out[i][1])+1);
				//applying ramp filter with origin in arraysize/2 and an maximum amplitude of 1
				filteredF[j + arraysize*i] = transformed[j + arraysize*i] * abs(arraysize / 2 - i)/(arraysize/2);
				
				//setting out for inverse transformation to filtered
				infftFiltered[i][0] = filteredF[j + arraysize*i];
				infftFiltered[i][1] = out[i][1];


				if (transformed[j + arraysize*i] > maxF) { maxF = transformed[j + arraysize*i]; }


				//cout << out[i][0] << endl;
			}
			fftw_execute(pBack);
			fftw_execute(pBackFiltered);
			
			//progress
			//cout <<"\t\t"<< (float) j / arraysize << endl;

			for (int i = 0; i < arraysize; i++) {

				filteredO[j + arraysize*i] = outfftFiltered[i][0]/arraysize;

				//setting and scaling the backtransformed image without filtering for clarity
				inverted[j + arraysize*i] = in[i][0] / arraysize;
				if (inverted[j + arraysize*i] > maxO) { maxO = inverted[j + arraysize*i]; }

			}
		}

		cout << "maxF = " << (float) maxF << endl;
		cout << "maxO = " << (float)maxO << endl;

		NRRD::ImageView<float> fourier(arraysize, arraysize,0, transformed);
		NRRD::ImageView<float> original(arraysize, arraysize, 0, inverted);
		NRRD::ImageView<float> filteredFImage(arraysize, arraysize, 0, filteredF);
		NRRD::ImageView<float> filteredOImage(arraysize, arraysize, 0, filteredO);

		//rif0->data() = filteredOImage;
		rif0->replaceRadonIntermediateData(filteredOImage);


		/*
		for (int j = 0; j < rif0->getRadonBinNumber(1); j++) {
			for (int i = 0; i < rif0->getRadonBinNumber(0); i++) {
				//if (cols0[j][i] != rif0->data().pixel(i, j)) {
					//cout << "Value at" << i << " " << j << endl;
				}					
			}
		}
		//fft.fwd(,);
		*/
		// Slow CPU evaluation for just two views
		// (usually, you just call ecc.evaluate(...) which uses the GPU to compute metric for all projections)
		EpipolarConsistency::MetricRadonIntermediate ecc(Ps,rifs);
		ecc.setdKappa(GetSet<double>    ("Epipolar Consistency/Sampling/Angle Step (deg)")/180*Geometry::Pi);
		double inconsistency=ecc.evaluateForImagePair(0,1, &redundant_samples0, &redundant_samples1,&kappas, &rif_samples0, &rif_samples1);
		
		// DEBUG ONLY
		//NRRD::ImageView<float> oneDim(, );
		//UtilsQt::Figure("check", oneDim, 0, 0, true);
		//UtilsQt::Figure("A",rif0->data(),0,0,true);
		//UtilsQt::Figure("B",rif1->data(),0,0,true);

		g_app.progressUpdate(6);

		// Visualize redundant signals
		QColor red(255,0,0);
		QColor black(0,0,0);
		Plot plot("ECC using Radon intermediate functions");
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

		// Show Radon intermediate functions.
		rif0->readback();
		Figure fig0("Radon Intermediate Function 0", rif0->data(),true);
		//rif1->readback();
		//Figure fig1("Radon Intermediate Function 1", rif1->data(),true);
		Figure fig2("Magnitude columnwise fourier transformed", fourier);
		Figure fig3("Backtransformed image", original);
		Figure fig4("Ramp filtered FFT", filteredFImage);
		Figure fig5("Backtransformed ramp-filtered image ", filteredOImage);


#ifndef DEBUG
		// Show sample locations (slow)
		using namespace std;
		bool check = rif0->isDerivative();
		double n_alpha2=0.5*rif0->data().size(0);
		double n_t2    =0.5*rif0->data().size(1);
		double step_alpha=rif1->getRadonBinSize(0);
		double step_t=rif1->getRadonBinSize(1);
		/*
		for (int i = 0; i < (int)rif_samples0.size(); i+=8) {
			fig0.drawPoint(rif_samples0[i].first / step_alpha + n_alpha2, rif_samples0[i].second / step_t + n_t2, black);
			//for (int i=0; i<(int)rif_samples1.size(); i++)
			//cout << rif_samples0[i].second / step_t + n_t2 << endl;
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
