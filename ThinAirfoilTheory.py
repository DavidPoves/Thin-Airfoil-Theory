import numpy as np
import os
import scipy.spatial as spatial
import scipy.integrate as integrate
import sympy as sp
import matplotlib.pyplot as plt


"""
MIT License

Copyright (c) 2020 David Poves

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


class ThinAirfoilTheory(object):
	def __init__(self):
		self.dat_file, self.txt_file = None, None
		self.x_coords_surf, self.y_coords_surf = np.array([]), np.array([])
		self.coords_surf_mat = np.array([])
		self.x_coords_mean, self.y_coords_mean = np.array([]), np.array([])
		self.coords_mean_mat = np.array([])
		self.poly_order = 5  # Set polynomial order to fit the mean line camber.
		self.aoa = None  # Preallocate the angle of attack.
		self.vor = None  # Object of the Voronoi diagram.
		self.best_pol_dict = dict()  # Dictionary containing the main information of the camber line's polynomial.
		self.best_pol = None
		self.best_pol_der = None  # First derivative of the best polynomial for mean camber line.
		self.sympy_poly_der = None  # Sympy polynomial of best_pol_der.
		self.sympy_der = None  # Sympy equation of sympy_poly_der with the change of variable required.
		self.lambda_der = None  # Lambda function of sympy_der.
		self.header = ''
		self.cl, self.zero_lift_angle = None, None

	@staticmethod
	def determine_best_polynomial_fit(x_data, y_data, max_degree=5):
		"""
		Determine the best polynomial to fit data based on the sum of the squared residuals of the least-squares fit.
		:param x_data: Array like of the x coordinates.
		:param y_data: Array like of data to be fitted.
		:param max_degree: Max. degree of polynomials to be considered.
		:return: Dictionary containing the best polynomial fit information.
		"""
		best_data = {'degree': 0, 'residual': 1e9, 'coefficients': []}
		for i in np.arange(1, max_degree + 1):
			data = np.polyfit(x_data, y_data, i, full=True)
			res = data[1][0]
			if res < best_data['residual']:
				best_data['degree'] = i
				best_data['residual'] = res
				best_data['coefficients'] = data[0]
		return best_data

	def read_file(self, filename=None):
		"""
		Read the file containing the x and y coordinates of the points used to discrete the pressure and suction
		surfaces. The file should be formatted such that the first row contains a header of the file (name of the
		airfoil or any other header info), and for the next rows the first column should contain the x coordinates and
		the second column should have the corresponding y coordinates. The user may provide the name of the .dat file to
		be read. Otherwise, the first .dat file contained in the current working directory will be loaded.
		The user may introduce .txt or .dat files. In case no filename is introduced, the program will search for .dat
		files only. In any possible case, the file will only be opened with read-only permissions.\n
		Once the file is properly loaded, the algorithm will automatically look for the x and y coordinates according to
		the specified format.\n
		:param filename: String of the filename. If the file is located outside the current working directory, the full
		path should be specified.
		:return: None
		"""
		# Process the filename.
		if filename is None:
			dat_file_lst = []
			for file in os.listdir(os.getcwd()):
				if file.endswith('.dat'):
					dat_file_lst.append(file)
			self.dat_file = dat_file_lst[0]
			filename = self.dat_file
		elif filename is not None:
			if isinstance(filename, str):  # We need to check if the file has the appropriate format.
				if filename.endswith('.dat'):
					self.dat_file = filename
				elif filename.endswith('.txt'):
					self.txt_file = filename
				else:  # Non-valid format was introduced.
					raise TypeError('The input data should be formatted as .dat or .txt!')

		# Open the file with read-only permissions.
		file_lines = []
		with open(filename, "r") as file:
			for line in file:
				file_lines.append(line)

		# Extract the x and y coordinates according to the specified format.
		self.header = ' '.join([element for element in file_lines[0].split()])  # Extract the header of the file.
		coords_arr = []
		for line in file_lines[1:]:  # Assuming the first line is the header (name of the airfoil).
			coord_lst = []
			for t in line.split():
				try:  # If the line contains any number(s), it will extract it/them.
					coord_lst.append(float(t))
				except ValueError:
					pass
			coords_arr.append(coord_lst)

		for coord in coords_arr:
			self.x_coords_surf = np.append(self.x_coords_surf, coord[0])
			self.y_coords_surf = np.append(self.y_coords_surf, coord[1])

		self.coords_surf_mat = np.column_stack((self.x_coords_surf, self.y_coords_surf))

	def visualize_voronoi_diagram(self):
		"""
		Method to let the user visualize how the Voronoi diagram computes the mean chamber line's set of points.
		:return:
		"""
		spatial.voronoi_plot_2d(self.vor)
		plt.plot()
		plt.scatter(self.x_coords_surf, self.y_coords_surf, label='surface')
		plt.plot(self.x_coords_mean, np.polyval(np.poly1d(self.best_pol_dict['coefficients']), self.x_coords_mean),
		         label='fit')
		plt.plot(self.x_coords_mean, self.y_coords_mean, label='real')
		plt.legend()

	def build_mean_camber_line(self, furthest_site=False, incremental=False, qhull_options=None, xlims=(0, 1),
	                           ylims=(-0.2, 0.2), max_pol_degree=7):
		"""
		Create the mean camber line of the loaded airfoil based on the loaded coordinates. Since in many cases the user
		may not have the equation of the mean camber line (or a set of points describing it), a method is required in
		order to automatically generate this set of points. To provide a general method for this purpose, the so-called
		Voronoi diagrams are used (https://en.wikipedia.org/wiki/Voronoi_diagram), wrapping the algorithm implemented in
		scipy. The reason to implement this process and not simply the average between the y coordinate of the upper and
		lower surfaces is that for most of the data sets available the x coordinates will not be the same for the
		pressure and suction sides, thus the average may not be accurate.\n
		In order to properly apply this algorithm resolution is of paramount importance. Hence, the user is encouraged
		to check the resolution of the surface. If at first the results are not quite satisfactory, the user is
		encouraged to use XFLR5, load the profile and refine both globally and locally.\n
		Once the points of the mean camber line are defined, these are later fitted using the best possible polynomial,
		up to degree 7.

		:param furthest_site: Whether to compute a furthest-site Voronoi diagram. Optional, default is False.
		:param incremental: Allow adding new points incrementally. This takes up some additional resources. Optional,
		default is False.
		:param qhull_options: Additional options to pass to Qhull. See Qhull manual for details. (Default: “Qbb Qc Qz Qx”
		for ndim > 4 and “Qbb Qc Qz” otherwise. Incremental mode omits “Qz”.)
		:param xlims: Array-like containing the limits for the Voronoi vertices. Only vertices located within this range
		will be selected. Optional, default is (0, 1) -> Normalized aerofoil.
		:param ylims: Array-like containing the limits for the Voronoi vertices. Only vertices located within this range
		will be selected. Optional, default is (-0.2, 0.2).
		:param max_pol_degree: Integer of the maximum degree to be used to fit the mean camber line.
		:return:
		"""
		# Build the Voronoi diagram.
		self.vor = spatial.Voronoi(self.coords_surf_mat, furthest_site=furthest_site, incremental=incremental,
		                           qhull_options=qhull_options)

		# Check which of the Voronoi vertices are within the desired range.
		assert len(xlims) == 2, f'Length of the x limits should be 2, not {len(xlims)}.'
		assert len(ylims) == 2, f'Length of the y limits should be 2, not {len(xlims)}.'
		for coord_vertices in self.vor.vertices:
			if xlims[0] <= coord_vertices[0] <= xlims[-1] and ylims[0] <= coord_vertices[1] <= ylims[1]:
				self.x_coords_mean = np.append(self.x_coords_mean, coord_vertices[0])
				self.y_coords_mean = np.append(self.y_coords_mean, coord_vertices[1])
			else:
				pass

		# Build a matrix with the coordinates to allow sorting the coordinates efficiently.
		self.coords_mean_mat = np.column_stack((self.x_coords_mean, self.y_coords_mean))
		self.coords_mean_mat = self.coords_mean_mat[self.coords_mean_mat[:, 0].argsort()]
		self.x_coords_mean = self.coords_mean_mat[:, 0]
		self.y_coords_mean = self.coords_mean_mat[:, 1]

		# Fit the points to the best possible polynomial.
		self.best_pol_dict = ThinAirfoilTheory.determine_best_polynomial_fit(self.x_coords_mean, self.y_coords_mean,
		                                                                     max_degree=max_pol_degree)
		self.best_pol = np.poly1d(self.best_pol_dict['coefficients'])
		self.best_pol_der = self.best_pol.deriv(1)

	def solve_theory(self, angle_of_attack, chord=1, n_coefficients=3, report=True):
		"""
		Solve the thin airfoil theory for a given angle of attack and chord length using the number of A coefficients
		as given in n_coefficients.
		:param angle_of_attack: Angle of attack of the airfoil IN RADIANS.
		:param chord: Chord length with same units as the x coordinates. Optional, default is normalized (c=1).
		:param n_coefficients: Number of coefficients to be solved. Must be greater than 2. Optional, default is 3.
		:param report: Boolean indicating if a report containing all relevant calculated information should be printed.
		:return: Tuple containing the corresponding coefficients.
		"""
		# Perform the change of variable. To do so, first transform the numpy polynomial to sympy.
		self.sympy_poly_der = sp.Poly(self.best_pol_der.coefficients, sp.Symbol('x'))

		# Perform the change of variable: x = c/2 * (1-cos(theta)).
		theta = sp.Symbol('theta')
		self.sympy_der = self.sympy_poly_der.subs({sp.Symbol('x'): 0.5*chord*(1-sp.cos(theta))})
		self.lambda_der = sp.lambdify(theta, self.sympy_der.as_expr(), modules='numpy')

		# Compute the coefficients.
		coefficients = []
		A0 = angle_of_attack - (1/np.pi)*integrate.quad(self.lambda_der, 0, np.pi)[0]
		coefficients.append(A0)
		assert n_coefficients >= 2, 'More than 1 coefficient should be computed in order to derive data from this theory'
		for i in np.arange(1, n_coefficients):
			coefficients.append((2/np.pi)*integrate.quad(lambda angle: self.lambda_der(angle)*np.cos(i*angle), 0, np.pi)
				[0])

		# Compute data derived from the theory.
		self._compute_relevant_data(coefficients, chord)

		# Print the report if required.
		if report:
			print("\n-----------------------------------------------------------------------------------\n")
			print(f"Showing the data derived from the Thin Airfoil Theory for profile {self.header}:\n")
			print(f"Angle of attack: {angle_of_attack*180/np.pi:.2f} degrees.\n")
			print(f"Zero lift angle of attack: {self.zero_lift_angle*180/np.pi:.2f} degrees.\n")
			print(f"Lift coefficient: {self.cl:.5f}\n")
			print(f"Moment coefficient around the leading edge: {self.cm_le:.5f}\n")
			print(f"Moment coefficient around the quarter chord: {self.cm_quarter:.5f}\n")
			print("\n-----------------------------------------------------------------------------------\n")

		return tuple(coefficients)

	def _compute_relevant_data(self, coefficients, chord):
		"""
		Compute all the information that can be derived from the thin airfoil theory given the necessary coefficients.
		:param coefficients: Array-like of all the computed coefficients.
		:param chord: The length of chord of the airfoil, with same units as the x coordinates.
		:return:
		"""
		# Compute the lift coefficient.
		self.cl = 2*np.pi*(coefficients[0] + 0.5*coefficients[1])

		# Compute the zero lift angle of attack.
		factor = -(1/np.pi)
		self.zero_lift_angle = factor*integrate.quad(lambda angle: self.lambda_der(angle)*(np.cos(angle)-1), 0, np.pi)[0]

		# Compute the moment coefficient about the Leading Edge.
		self.cm_le = -(self.cl/4 + np.pi/4*(coefficients[1] - coefficients[2]))

		# Compute the moment coefficient about the quarter chord.
		self.cm_quarter = np.pi/4 * (coefficients[2] - coefficients[1])

		# Compute the center of pressure.
		self.x_cp = chord/4*(1+np.pi/self.cl*(coefficients[1] - coefficients[2]))


if __name__ == '__main__':
	thin_theory = ThinAirfoilTheory()
	thin_theory.read_file(filename='NACA 2408.dat')
	thin_theory.build_mean_camber_line(max_pol_degree=7)
	A0, A1, A2 = thin_theory.solve_theory(angle_of_attack=5*np.pi/180, n_coefficients=3)
