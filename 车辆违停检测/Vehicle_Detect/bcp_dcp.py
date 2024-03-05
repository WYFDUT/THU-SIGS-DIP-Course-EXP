import cv2
import numpy as np

def dark_channel_prior(img, window_size):
    # Compute the dark channel of the image
    min_channel = np.minimum(np.minimum(img[:, :, 0], img[:, :, 1]), img[:, :, 2])
    dark_channel = cv2.erode(min_channel, np.ones((window_size, window_size)))
    return dark_channel

def atmospheric_light(img, dark_channel):
    # Select the top 0.1% brightest pixels from the dark channel
    flat_dark_channel = dark_channel.flatten()
    bright_pixels = np.argsort(flat_dark_channel)[-int(0.001 * len(flat_dark_channel)):]

    # Estimate atmospheric light as the maximum intensity in the selected pixels
    atmospheric_light = np.max(np.max(img, axis=2).flatten()[bright_pixels])
    return atmospheric_light

def transmission(img, atmospheric_light, omega=0.95, window_size=15):
    # Estimate the transmission map
    normalized_img = img / atmospheric_light
    dark_channel = dark_channel_prior(normalized_img, window_size)
    transmission = 1 - omega * dark_channel
    return transmission

def enhance_image(img, transmission, atmospheric_light, t0=0.1):
    # Ensure transmission map and atmospheric light have compatible shapes
    transmission = np.expand_dims(transmission, axis=2)

    # Clamp the transmission to avoid division by zero
    transmission = np.maximum(transmission, t0)

    # Recover the scene radiance
    enhanced_image = ((img.astype(np.float32) - atmospheric_light) / transmission) + atmospheric_light
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)

    return enhanced_image

def clahe(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab_image[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahed_l_channel = clahe.apply(l_channel)
    lab_image[:, :, 0] = clahed_l_channel
    clahed_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
    return clahed_image

def enhance(img):
    # Set parameters
    window_size = 15
    omega = 0.95
    t0 = 0.1
    # Apply the dark channel prior low light enhancement
    dark_channel = dark_channel_prior(img, window_size)
    atm_light = atmospheric_light(img, dark_channel)
    transmission_map = transmission(img, atm_light, omega, window_size)
    enhanced_image = enhance_image(img, transmission_map, atm_light, t0)
    enhanced_image = clahe(img)
    cv2.imwrite("test_new.png", enhanced_image)
    return enhanced_image


