# Thin-Airfoil-Theory

## Description.
Implement the Thin Airfoil Theory given the coordinates of the profile to be analysed. The format of the .dat (or .txt) file should contain as the first line the header, and the rest of the rows should contain two columns, where the first one contains the x coordinates and the last one the corresponding y coordinates.

## Algorithm
Since the necessary equation of the mean camber line is not available for many complex profiles, the thin airfoil theory cannot be applied directly. Moreover, the x coordinates of the upper and lower surfaces will not coincide in most of the available datasets, thus the average between y coordinates may not yield exact results. Consequently, a more general approach should be implemented. To achieve the former, the so called **Voronoi diagrams** are implemented (https://en.wikipedia.org/wiki/Voronoi_diagram) using the *scipy* algorithm. As a result, a set of points belonging to the mean camber line are obtained. Finally, a fit of this data is performed using polynomials, which result in quite accurate fittings for most of the airfoils. The used degree of the polynomial is automatically chosen based on the residuals (sum of the squares of the errors). Finally, the analytical equations are implemented, yielding as many coefficients as the user desires.

## Troubleshooting.
The user may visualize how good the actual fitting using the built-in method `visualize_voronoi_diagram`. In case the resulting set of points (labeled as *real* on the generated graph) oscillates a lot in some regions, try increasing the number of coordinates. To do so, one may load the profile on *XFLR5* and refine it both globally and locally where required. The accuracy of the *Voronoi* diagram vastly depends on the resolution of the profile. In case the polynomial fitting is not good enough, the user is encouraged to implement its own fitting algorithm.

## References.
Anderson Jr, J. D. (2010). Fundamentals of aerodynamics. Tata McGraw-Hill Education.
