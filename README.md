# Record-Image-Transformations
## Aim
To perform image transformations such as **Translation, Scaling, Shearing, Reflection, Rotation, and Cropping** using **OpenCV** and **Python**.

---

## Software Required
- **Anaconda** – Python 3.7  
- **OpenCV (cv2)** library  
- **NumPy** library  

## Algorithm

**Step 1:**  
Import the necessary libraries such as `cv2` and `numpy`.

**Step 2:**  
Read the input image using `cv2.imread()` and display the original image using `cv2.imshow()`.

**Step 3:**  
Perform various geometric transformations:  
- **Translation:** Shift the image position using a transformation matrix.  
- **Scaling:** Resize the image using `cv2.resize()`.  
- **Shearing:** Apply an affine transformation to skew the image.  
- **Reflection:** Flip the image using `cv2.flip()`.  
- **Rotation:** Rotate the image using `cv2.getRotationMatrix2D()` and `cv2.warpAffine()`.  
- **Cropping:** Slice the image array to extract a specific region.

**Step 4:**  
Display all the transformed images in separate windows using `cv2.imshow()`.

**Step 5:**  
Wait for a key press using `cv2.waitKey(0)` and then close all OpenCV windows using `cv2.destroyAllWindows()`.

## Program 

```python
Developed By : EASWAR R
Reg No : 212223230053

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Chennai_Central.jpg')
image.shape

#
# Display the images.
plt.imshow(image[:,:,::-1])
plt.title('Original Image')
plt.show()


# i) Image Translation
tx, ty = 100, 200  # Translation factors (shift by 100 pixels horizontally and 50 vertically)
M_translation = np.float32([[1, 0, tx], [0, 1, ty]])  # Translation matrix: 
# [1, 0, tx] - Horizontal shift by tx
# [0, 1, ty] - Vertical shift by ty
translated_image = cv2.warpAffine(image, M_translation, (636, 438)) 


plt.imshow(translated_image[:,:,::-1])
plt.title("Translated Image")
plt.axis('on')
plt.show()

# Image Scaling
fx, fy = 2.0, 1.0  
scaled_image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

plt.imshow(scaled_image[:,:,::-1]) 
plt.title("Scaled Image") 
plt.axis('on')
plt.show()

# Image Shearing
shear_matrix = np.float32([[1, 0.5, 0], [0.5, 1, 0]])  
sheared_image = cv2.warpAffine(image, shear_matrix, (636, 438))

plt.imshow(sheared_image[:,:,::-1])
plt.title("Sheared Image") 
plt.axis('on')
plt.show()

# Image Reflection
reflected_image = cv2.flip(image, 2)  # Flip the image horizontally (1 means horizontal flip)

# flip: 1 means horizontal flip, 0 would be vertical flip, -1 would flip both axes

# Show original image 
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image[:, :, ::-1])
plt.title("Original Image")
plt.axis('off')

# Show reflected image 
plt.subplot(1, 2, 2)
plt.imshow(reflected_image[:,:,::-1])
plt.title("Reflected Image")
plt.axis('off')

plt.tight_layout()
plt.show()

# Image Rotation
(height, width) = image.shape[:2]  # Get the image height and width
angle = 45  # Rotation angle in degrees (rotate by 45 degrees)
center = (width // 2, height // 2)  # Set the center of rotation to the image center
M_rotation = cv2.getRotationMatrix2D(center, angle, 1)  # Get the rotation matrix
# getRotationMatrix2D: Takes the center of rotation, angle, and scale factor (1 means no scaling)
rotated_image = cv2.warpAffine(image, M_rotation, (width, height))  # Apply rotation

plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))  # Display the rotated image
plt.title("Rotated Image")  # Set title
plt.axis('off')


# Image Rotation
image .shape    # height, width, channel 
angle = 145  
center = (636 // 2, 438 // 2)  
M_rotation = cv2.getRotationMatrix2D(center, angle, 1)  
# getRotationMatrix2D: Takes the center of rotation, angle, and scale factor (1 means no scaling)
rotated_image = cv2.warpAffine(image, M_rotation, (width, height)) 

plt.imshow(rotated_image[:,:,::-1])  # Display the rotated image
plt.title("Rotated Image")  # Set title
plt.axis('off')
plt.show()

# Image Cropping
x, y, w, h = 0, 0, 200, 150  

cropped_image = image[y:y+h, x:x+w]   # Format: image[start_row:end_row, start_col:end_col]

# Show original image 
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image[:, :, ::-1])
plt.title("Original Image")
plt.axis('on')

# Show reflected image 
plt.subplot(1, 2, 2)
plt.imshow(cropped_image[:,:,::-1])
plt.title("Cropped Image")
plt.axis('on')

plt.tight_layout()
plt.show()
```
