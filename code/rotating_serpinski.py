

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Function to generate Sierpiński triangles recursively using interpolation
def sierpinski_interpolated(vertices, level, t):
    """Recursively generates Sierpiński triangles using (1-t)x + t*y interpolation."""
    if level == 0:
        return [vertices]

    # Compute new points using the interpolation rule
    new_pts = np.array([
        (1 - t) * vertices[0] + t * vertices[1],
        (1 - t) * vertices[1] + t * vertices[2],
        (1 - t) * vertices[2] + t * vertices[0]
    ])

    # Generate recursively subdivided sub-triangles
    triangles = []
    alternating_rotation = False
    triangles += sierpinski_interpolated(np.array([vertices[0], new_pts[0], new_pts[2]]), level - 1, 1-t if alternating_rotation else t)
    triangles += sierpinski_interpolated(np.array([vertices[1], new_pts[1], new_pts[0]]), level - 1, 1-t if alternating_rotation else t)
    triangles += sierpinski_interpolated(np.array([vertices[2], new_pts[2], new_pts[1]]), level - 1, 1-t if alternating_rotation else t)
    
    return triangles


def main():
	# Parameters
	depth = 7  # Depth of recursion

	# Initial equilateral triangle vertices
	main_triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])

	# Set up figure
	fig, ax = plt.subplots(figsize=(6, 6))
	plt.subplots_adjust(bottom=0.2)  # Space for slider
	ax.set_xlim(-0.1, 1.1)
	ax.set_ylim(-0.1, 1.1)
	ax.set_xticks([])
	ax.set_yticks([])

	# Initial plot
	t_init = 0.5  # Starting value of t
	triangles = sierpinski_interpolated(main_triangle, depth, t_init)
	triangle_lines = [ax.plot(*np.append(tri, [tri[0]], axis=0).T, '-', color='blue', alpha=0.7)[0] for tri in triangles]

	# Slider setup
	ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])  # Position of the slider
	t_slider = Slider(ax_slider, 't', 0, 1, valinit=t_init)

	# Update function for slider
	def update(val):
		t = t_slider.val  # Get slider value
		new_triangles = sierpinski_interpolated(main_triangle, depth, t)
		
		# Update each line plot
		for line, tri in zip(triangle_lines, new_triangles):
			line.set_data(*np.append(tri, [tri[0]], axis=0).T)
		
		fig.canvas.draw_idle()  # Redraw the figure

	# Connect slider to update function
	t_slider.on_changed(update)

	# Show interactive plot
	plt.show()

if __name__ == '__main__':
	main()
