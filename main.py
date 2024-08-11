import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import io
from PIL import Image
import cv2

def plot_data(data):
    fig, ax = plt.subplots(figsize=(10, 10))
    for paths in data:
        for path in paths:
            x, y = path.T
            ax.plot(x, y, marker='o')
    ax.set_aspect('equal')

    # Remove axes and ticks
    ax.axis('off')

    # Remove the frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Remove any remaining ticks
    ax.tick_params(axis='both', which='both', length=0)

    # Save the figure to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()
    return buf

def read_csv_to_path_XYs(csv_content):
    np_path_XYs = np.genfromtxt(io.StringIO(csv_content.decode('utf-8')), delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs


def save_image(img_buffer, filename):
    # Create 'images' folder if it doesn't exist
    if not os.path.exists('images'):
        os.makedirs('images')
    
    # Save the image
    img_path = os.path.join('images', filename)
    with open(img_path, 'wb') as f:
        f.write(img_buffer.getvalue())
    return img_path

def save_image_symmetry(img_buffer, filename):
    # Create 'images' folder if it doesn't exist
    if not os.path.exists('symmetry_images'):
        os.makedirs('symmetry_images')
    
    # Save the image
    img_path = os.path.join('symmetry_images', filename)
    with open(img_path, 'wb') as f:
        f.write(img_buffer.getvalue())
    return img_path

def save_image_symmetry_points(img_buffer, filename):
    # Create 'images' folder if it doesn't exist
    if not os.path.exists('points_images'):
        os.makedirs('points_images')
    
    # Save the image
    img_path = os.path.join('points_images', filename)
    with open(img_path, 'wb') as f:
        f.write(img_buffer.getvalue())
    return img_path



def detect_and_regularize_shapes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    canvas = image.copy()  # Use a copy of the original image instead of a blank canvas

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        if len(approx) == 3:
            shape = "triangle"
            cv2.drawContours(canvas, [approx], 0, (0, 255, 0), 2)

        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                shape = "square"
            else:
                shape = "rectangle"
            cv2.drawContours(canvas, [approx], 0, (0, 0, 255), 2)

        elif len(approx) == 5:
            shape = "pentagon"
            cv2.drawContours(canvas, [approx], 0, (255, 0, 0), 2)

        elif len(approx) == 6:
            shape = "hexagon"
            cv2.drawContours(canvas, [approx], 0, (255, 255, 0), 2)

        elif len(approx) > 6:
            # Check for circle or ellipse
            area = cv2.contourArea(contour)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.8:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(canvas, center, radius, (0, 255, 255), 2)
                shape = "circle"
            else:
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(canvas, ellipse, (255, 0, 255), 2)
                shape = "ellipse"

        else:
            shape = "unknown"
            cv2.drawContours(canvas, [contour], 0, (128, 128, 128), 2)

        # Add shape label
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(canvas, shape, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return canvas

def process_images(image_paths):
    image_names = []
    results = {}

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load the image {image_path}.")
            continue

        image_name = os.path.basename(image_path)
        image_names.append(image_name)
        output_image = detect_and_regularize_shapes(image)
        results[image_name] = {'processed_image': output_image}

    return image_names, results


def display_images(image_paths, results):
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        output_image = results[image_name]['processed_image']
        original_image = cv2.imread(image_path)

        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.imshow(original_image_rgb)
        ax1.set_title(f'Original {image_name}')
        ax1.axis('off')

        ax2.imshow(output_image_rgb)
        ax2.set_title(f'Regularized Shapes {image_name}')
        ax2.axis('off')

        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Display in Streamlit
        st.image(buf, caption=f"Comparison for {image_name}", use_column_width=True)
        
        # Save processed image
        output_directory = 'output'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        output_path = os.path.join(output_directory, f"processed_{image_name}")
        cv2.imwrite(output_path, output_image)
        
        # Provide download button for processed image
        with open(output_path, "rb") as file:
            st.download_button(
                label=f"Download Processed {image_name}",
                data=file,
                file_name=f"processed_{image_name}",
                mime="image/png"
            )
            
def check_horizontal_symmetry(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Unable to read the image file {image_path}")
        return

    img_symmetry = img.copy()
    cv2.line(img_symmetry, (0, img.shape[0] // 2), (img.shape[1], img.shape[0] // 2), (255, 0, 255), 5)

    return img, img_symmetry

def check_vertical_symmetry(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Unable to read the image file {image_path}")
        return

    img_symmetry = img.copy()
    cv2.line(img_symmetry, (img.shape[1] // 2, 0), (img.shape[1] // 2, img.shape[0]), (255, 0, 255), 5)

    return img, img_symmetry

def process_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Unable to read the image file {image_path}")
        return

    return img


def get_symmetry():
  dataset_directory = './symmetry_images'
  image_files = [f for f in os.listdir(dataset_directory) if f.endswith(('.jpg', '.png', '.jpeg'))]
  
  fig = plt.figure(figsize=(20, 5 * len(image_files)))
  
  for i, image_file in enumerate(image_files):
      image_path = os.path.join(dataset_directory, image_file)
  
      print(f"Processing image: {image_file}")
  
      # Process the image
      original = process_image(image_path)
  
      # Check vertical symmetry
      _, vertical_symmetry = check_vertical_symmetry(image_path)
  
      # Check horizontal symmetry
      _, horizontal_symmetry = check_horizontal_symmetry(image_path)
  
      # Display images
      ax1 = fig.add_subplot(len(image_files), 3, i*3 + 1)
      ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
      ax1.set_title('Original')
      ax1.axis('off')
  
      ax2 = fig.add_subplot(len(image_files), 3, i*3 + 2)
      ax2.imshow(cv2.cvtColor(vertical_symmetry, cv2.COLOR_BGR2RGB))
      ax2.set_title('Vertical Symmetry')
      ax2.axis('off')
  
      ax3 = fig.add_subplot(len(image_files), 3, i*3 + 3)
      ax3.imshow(cv2.cvtColor(horizontal_symmetry, cv2.COLOR_BGR2RGB))
      ax3.set_title('Horizontal Symmetry')
      ax3.axis('off')
  
      plt.tight_layout()
      st.pyplot(fig)

        # Save processed images
      output_directory = 'output_symmetry'
      if not os.path.exists(output_directory):
          os.makedirs(output_directory)
      
      for img, suffix in [(vertical_symmetry, 'vertical'), (horizontal_symmetry, 'horizontal')]:
          output_path = os.path.join(output_directory, f"{suffix}_symmetry_{os.path.basename(image_file)}")
          cv2.imwrite(output_path, img)
          
          # Provide download button for processed image
          with open(output_path, "rb") as file:
              st.download_button(
                  label=f"Download {suffix.capitalize()} Symmetry Image",
                  data=file,
                  file_name=f"{suffix}_symmetry_{os.path.basename(image_file)}",
                  mime="image/png"
              )
          

def find_symmetry_points(contour):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    symmetry_axes = []
    n = len(approx)
    for i in range(n):
        for j in range(i+1, n):
            pt1, pt2 = tuple(approx[i][0]), tuple(approx[j][0])
            dist_to_center = np.abs((pt2[1]-pt1[1])*cX - (pt2[0]-pt1[0])*cY + pt2[0]*pt1[1] - pt2[1]*pt1[0]) / np.sqrt((pt2[1]-pt1[1])**2 + (pt2[0]-pt1[0])**2)
            if dist_to_center < 5:
                symmetry_axes.append((pt1, pt2))

    intersection_points = set()
    for axis in symmetry_axes:
        for i in range(len(contour)):
            pt1, pt2 = tuple(contour[i][0]), tuple(contour[(i+1) % len(contour)][0])
            intersection = line_intersection(axis[0], axis[1], pt1, pt2)
            if intersection:
                intersection_points.add(intersection)

    return list(intersection_points)

def line_intersection(line1_pt1, line1_pt2, line2_pt1, line2_pt2):
    x1, y1 = line1_pt1
    x2, y2 = line1_pt2
    x3, y3 = line2_pt1
    x4, y4 = line2_pt2

    denominator = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denominator == 0:
        return None

    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denominator
    u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denominator

    if 0 <= t <= 1 and 0 <= u <= 1:
        x = int(x1 + t*(x2-x1))
        y = int(y1 + t*(y2-y1))
        return (x, y)
    return None


def process_image_points(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to gray-scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the colours
    gray_inverted = cv2.bitwise_not(gray)

    # Create a binary thresholded image
    _, binary = cv2.threshold(gray_inverted, 100, 255, cv2.THRESH_BINARY)

    # Find the contours from the thresholded image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Create a white image for drawing
    result_image = np.ones_like(image) * 255

    # Draw smooth contours and symmetry points
    for contour in contours:
        # Draw the smooth contour
        cv2.drawContours(result_image, [contour], 0, (0, 0, 0), 2)  # Black contour

        # Find and draw symmetry points
        symmetry_points = find_symmetry_points(contour)
        for point in symmetry_points:
            cv2.circle(result_image, point, 5, (0, 0, 255), -1)  # Red symmetry points

    return result_image

def points_symmetry():
    examples_path = "./points_images"
    
    st.title("Point Symmetry Image Processing")
    
    # Get all PNG files in the examples directory
    png_files = [f for f in os.listdir(examples_path) if f.endswith('.png')]

    if not png_files:
        st.write(f"No PNG files found in {examples_path}. Please upload PNG files to this folder.")
    else:
        st.write(f"Found {len(png_files)} PNG files in {examples_path}.")
    
        # Process each PNG file and display
        for png_file in png_files:
            image_path = os.path.join(examples_path, png_file)
            result_image = process_image_points(image_path)
    
            # Display the result in Streamlit
            st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), caption=f"Smooth Contours with Symmetry Points: {png_file}", use_column_width=True)

    st.write("Processing complete.")
            
    
    print("Processing complete.")

def main():
    st.title("CSV to PNG Converter")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="regularization")

    if uploaded_file is not None:
        # Read the CSV file
        csv_content = uploaded_file.read()
        
        # Convert CSV to path_XYs
        path_XYs = read_csv_to_path_XYs(csv_content)
        
        # Plot the data and get the image buffer
        img_buffer = plot_data(path_XYs)
        
        # Generate a filename for the image
        filename = f"{uploaded_file.name.rsplit('.', 1)[0]}.png"
        
        # Save the image to the 'images' folder
        saved_path = save_image(img_buffer, filename)
        
        dataset_directory = "./images"
        image_files = [os.path.join(dataset_directory, f) for f in os.listdir(dataset_directory) if f.endswith(('.jpg', '.png', '.jpeg'))]

        image_names, results = process_images(image_files)
        display_images(image_files, results)
        
        
        # Display the image
        st.image(img_buffer, caption="Generated PNG Image", use_column_width=True)
        
        # Provide a download button for the PNG
        st.download_button(
            label="Download PNG",
            data=img_buffer,
            file_name=filename,
            mime="image/png"
        )
        
        # Provide download buttons for both images
        with open(saved_path, "rb") as file:
            st.download_button(
                label="Download Original PNG",
                data=file,
                file_name=filename,
                mime="image/png"
            )
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="symmetry")
    
    if uploaded_file is not None:
        csv_content = uploaded_file.read()
        path_XYs = read_csv_to_path_XYs(csv_content)
        
        # Plot the data and get the image buffer
        img_buffer = plot_data(path_XYs)
        
        # Generate a filename for the image
        filename = f"{uploaded_file.name.rsplit('.', 1)[0]}.png"
        
        saved_path = save_image_symmetry(img_buffer, filename)
        
        dataset_directory = "./symmetry_images"
        image_files = [os.path.join(dataset_directory, f) for f in os.listdir(dataset_directory) if f.endswith(('.jpg', '.png', '.jpeg'))]
        get_symmetry()
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="symmetry_points")
    
    if uploaded_file is not None:
        csv_content = uploaded_file.read()
        path_XYs = read_csv_to_path_XYs(csv_content)
        
        # Plot the data and get the image buffer
        img_buffer = plot_data(path_XYs)
        
        # Generate a filename for the image
        filename = f"{uploaded_file.name.rsplit('.', 1)[0]}.png"
        
        saved_path = save_image_symmetry_points(img_buffer, filename)
        
        dataset_directory = "./points_images"
        image_files = [os.path.join(dataset_directory, f) for f in os.listdir(dataset_directory) if f.endswith(('.jpg', '.png', '.jpeg'))]
        points_symmetry()
        
        
        

if __name__ == "__main__":
    main()
