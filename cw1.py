import numpy as np
import matplotlib.pyplot as plt

def generate_horizontal_plane(width, length, num_points):
    x = np.random.uniform(0, width, num_points)
    y = np.random.uniform(0, length, num_points)
    z = np.zeros(num_points)
    return np.column_stack((x, y, z))

def generate_vertical_plane(width, height, num_points):
    x = np.random.uniform(0, width, num_points)
    y = np.zeros(num_points)
    z = np.random.uniform(0, height, num_points)
    return np.column_stack((x, y + 30, z))  # przesunięcie w osi Y

def generate_cylinder(radius, height, num_points):
    theta = np.random.uniform(0, 2*np.pi, num_points)
    z = np.random.uniform(0, height, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta) + 60  # przesunięcie w osi Y
    return np.column_stack((x, y, z))

def save_to_xyz(points, filename):
    np.savetxt(filename, points, fmt="%.6f")

def visualize_all(plane_horizontal, plane_vertical, cylinder):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(plane_horizontal[:,0], plane_horizontal[:,1], plane_horizontal[:,2], s=1, label='Horizontal Plane')
    ax.scatter(plane_vertical[:,0], plane_vertical[:,1], plane_vertical[:,2], s=1, label='Vertical Plane')
    ax.scatter(cylinder[:,0], cylinder[:,1], cylinder[:,2], s=1, label='Cylinder Surface')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Generated Point Clouds')
    ax.legend()
    plt.show()

def generate_and_save_clouds():
    num_points = 1000

    plane_horizontal = generate_horizontal_plane(width=10, length=15, num_points=num_points)
    plane_vertical = generate_vertical_plane(width=10, height=20, num_points=num_points)
    cylinder = generate_cylinder(radius=5, height=20, num_points=num_points)

    all_points = np.vstack((plane_horizontal, plane_vertical, cylinder))
    save_to_xyz(all_points, "all_clouds.xyz")

    visualize_all(plane_horizontal, plane_vertical, cylinder)

if __name__ == "__main__":
    generate_and_save_clouds()