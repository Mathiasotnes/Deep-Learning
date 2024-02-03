import numpy as np
import math
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random


class Generator():

    def __init__(self,
                 image_size=30,
                 total_images=100,
                 num_classes=4,
                 shape_types=None,
                 max_object_size_ratio=2,
                 noise_level=0.1,
                 dataset_split=(0.7, 0.2, 0.1),
                 flatten=False):
        self.image_size = image_size
        self.total_images = total_images
        self.num_classes = num_classes
        self.shape_types = shape_types if shape_types else [
            'circle', 'rectangle', 'triangle', 'cross'
        ]
        self.max_object_size_ratio = max_object_size_ratio
        self.noise_level = noise_level
        self.dataset_split = dataset_split
        self.flatten = flatten

    def generate(self, noise_level=None, one_hot_labels=True):
        images, labels = [], []
        # Generate images for each class
        for label, shape in enumerate(self.shape_types):
            num_images = self.total_images // self.num_classes
            for _ in range(num_images):
                image = self._generate_single_image(shape, noise_level)
                images.append(image)
                labels.append(label)
        
        images = np.array(images)
        labels = np.array(labels)

        if one_hot_labels:
            labels = np.eye(self.num_classes)[labels]

        # Split dataset
        train_images, train_labels, val_images, val_labels, test_images, test_labels = self._split_dataset(
            images, labels)

        if self.flatten:
            train_images = train_images.reshape(train_images.shape[0], -1)
            val_images = val_images.reshape(val_images.shape[0], -1)
            test_images = test_images.reshape(test_images.shape[0], -1)

        return (train_images, train_labels), (val_images,
                                              val_labels), (test_images,
                                                            test_labels)

    def visualize_images(self, images, labels=None):
        num_images = len(images)
        num_cols = 4  # You can adjust this based on your preference
        num_rows = num_images // num_cols + (1 if num_images % num_cols else 0)

        plt.figure(figsize=(10, 10))
        for i in range(num_images):
            plt.subplot(num_rows, num_cols, i + 1)
            if self.flatten:
                plt.imshow(images[i].reshape(self.image_size, self.image_size),
                           cmap='gray')
            else:
                plt.imshow(images[i], cmap='gray')
            plt.axis('off')
            if labels is not None:
                if labels.ndim > 1:
                    label = np.argmax(labels[i])
                    shape_name = self.shape_types[label]
                    plt.title(f'{shape_name}')
                else:
                    shape_name = self.shape_types[labels[i]]
                    plt.title(f'{shape_name}')
        plt.tight_layout()
        plt.show()

    def _generate_single_image(self, shape, temperature):
        # Create a blank image
        image = Image.new('L', (self.image_size, self.image_size), 0)
        draw = ImageDraw.Draw(image)

        # Calculate max object size based on the image size
        max_object_size = self.image_size // self.max_object_size_ratio

        # Draw the specified shape
        if shape == 'circle':
            self._draw_circle(draw, max_object_size)

        elif shape == 'rectangle':
            self._draw_rectangle(draw, max_object_size)

        elif shape == 'triangle':
            self._draw_triangle(draw, max_object_size)

        elif shape == 'cross':
            self._draw_cross(draw, max_object_size)

        # Convert to a numpy array
        np_image = np.array(image)

        # Add noise based on the temperature
        np_image = self._add_noise(np_image, temperature)

        return np_image

    def _add_noise(self, image, temperature):
        # Generate Gaussian noise
        mean = 0
        sigma = temperature  # Standard deviation of the noise
        gaussian_noise = np.random.normal(mean, sigma, image.shape)

        # Add the noise to the image
        noisy_image = image + gaussian_noise

        # Clip the values to stay within the valid range [0, 255] for images
        noisy_image = np.clip(noisy_image, 0, 255)

        return noisy_image

    def _split_dataset(self, images, labels):
        # Shuffle images and labels
        indices = np.arange(images.shape[0])
        np.random.shuffle(indices)
        images, labels = images[indices], labels[indices]

        # Calculate split indices
        train_end = int(len(images) * self.dataset_split[0])
        val_end = train_end + int(len(images) * self.dataset_split[1])

        # Split the dataset
        train_images, train_labels = images[:train_end], labels[:train_end]
        val_images, val_labels = images[train_end:val_end], labels[
            train_end:val_end]
        test_images, test_labels = images[val_end:], labels[val_end:]

        return train_images, train_labels, val_images, val_labels, test_images, test_labels

    def _draw_circle(self, draw, max_object_size):
        # Randomize the radius of the circle
        radius = random.randint(5, max_object_size // 2)

        # Randomize the center of the circle
        x_center = random.randint(radius, self.image_size - radius)
        y_center = random.randint(radius, self.image_size - radius)

        # Calculate the bounding box for the circle
        left_up_point = (x_center - radius, y_center - radius)
        right_down_point = (x_center + radius, y_center + radius)

        # Draw the circle
        draw.ellipse([left_up_point, right_down_point], fill=255)

    def _draw_rectangle(self, draw, max_object_size):
        # Randomize the dimensions of the rectangle
        rect_width = random.randint(5, max_object_size)
        rect_height = random.randint(5, max_object_size)

        # Randomize the top-left corner of the rectangle
        left_x = random.randint(0, self.image_size - rect_width)
        top_y = random.randint(0, self.image_size - rect_height)

        # Calculate the bottom-right corner of the rectangle
        right_x = left_x + rect_width
        bottom_y = top_y + rect_height

        # Draw the rectangle
        draw.rectangle([left_x, top_y, right_x, bottom_y], fill=255)

    def _calculate_angle(self, p0, p1, p2):
        """Calculate the angle at p1 formed by the line segments p0p1 and p1p2."""
        a = np.array(p0)
        b = np.array(p1)
        c = np.array(p2)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba,
                              bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return angle

    def _draw_triangle(self, draw, max_object_size):
        min_angle = 10 * math.pi / 180  # 10 degrees in radians

        while True:
            # Generate three random points
            points = [(random.randint(0, self.image_size),
                       random.randint(0, self.image_size)) for _ in range(3)]

            # Calculate angles at each vertex
            angle1 = self._calculate_angle(points[0], points[1], points[2])
            angle2 = self._calculate_angle(points[1], points[2], points[0])
            angle3 = self._calculate_angle(points[2], points[0], points[1])

            # Check if all angles are above the threshold
            if angle1 > min_angle and angle2 > min_angle and angle3 > min_angle:
                break

        # Draw the triangle
        draw.polygon(points, fill=255)

    def _draw_cross(self, draw, max_object_size):
        # Randomize the size and position of the cross
        cross_size = random.randint(5, max_object_size)
        center_x = random.randint(cross_size // 2,
                                  self.image_size - cross_size // 2)
        center_y = random.randint(cross_size // 2,
                                  self.image_size - cross_size // 2)

        # Calculate the end points of the two lines
        # Vertical line
        top_point = (center_x, center_y - cross_size // 2)
        bottom_point = (center_x, center_y + cross_size // 2)
        # Horizontal line
        left_point = (center_x - cross_size // 2, center_y)
        right_point = (center_x + cross_size // 2, center_y)

        # Draw the two lines
        draw.line([top_point, bottom_point], fill=255)
        draw.line([left_point, right_point], fill=255)
