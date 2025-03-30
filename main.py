import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def extract_bounding_box_mask(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((100, 100), np.uint8)
    dilated = cv2.dilate(thresholded, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    height, width = gray.shape
    filtered_contours = []
    for contour in contours:
        if not np.any(contour == 0) and not np.any(contour[:, :, 0] == width - 1) and not np.any(
                contour[:, :, 1] == height - 1):
            filtered_contours.append(contour)


    combined_contour = np.vstack(filtered_contours)


    x, y, w, h = cv2.boundingRect(combined_contour)


    mask = np.zeros_like(gray)


    mask[y:y + h, x:x + w] = 255

    return mask

    return mask

def extract_cup(image_path, disc_mask):

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None


    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    masked_image = cv2.bitwise_and(gray_image, gray_image, mask=disc_mask)

    equalized_image = cv2.equalizeHist(masked_image)


    _, thresholded = cv2.threshold(equalized_image, 245, 255, cv2.THRESH_BINARY)


    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    ellipse_mask = np.zeros_like(equalized_image, dtype=np.uint8)
    if len(contours) > 0:

        unified_contour = np.vstack(contours)

        if len(unified_contour) >= 5:
            ellipse = cv2.fitEllipse(unified_contour)
            (x, y), (major_axis, minor_axis), angle = ellipse

            # Calcular el área de la elipse
            ellipse_area = np.pi * (major_axis / 2) * (minor_axis / 2)

            if ellipse_area < 860:
                scale_factor = 2.8
            else:
                scale_factor = 2.1

            x_adjusted = x - major_axis * 0.5
            major_axis_adjusted = major_axis * scale_factor

            if minor_axis/major_axis_adjusted < 0.8:
                minor_axis = major_axis_adjusted * 0.9

            if major_axis_adjusted > 10 and minor_axis > 10:
                adjusted_ellipse = ((x_adjusted, y), (major_axis_adjusted, minor_axis), angle)
                cv2.ellipse(ellipse_mask, adjusted_ellipse, 255, -1)

    return ellipse_mask


def extract_disc(image_path, max_iterations=5):

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None
    else:
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    _, _, red_channel = cv2.split(image)

    kernel = create_circular_kernel(7)
    equalized_red = cv2.equalizeHist(red_channel)
    _, thresholded_image = cv2.threshold(equalized_red, 245, 255, cv2.THRESH_BINARY)
    opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)

    contours_disc, _ = cv2.findContours(opened_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours_disc:
        biggest_contour = max(contours_disc, key=cv2.contourArea)
    else:
        biggest_contour = None

    ellipse_mask = np.zeros_like(red_channel, dtype=np.uint8)

    iterations = 0
    while iterations<2:

        ellipse = cv2.fitEllipse(biggest_contour)
        (x, y), (major_axis, minor_axis), angle = ellipse

        aspect_ratio = min(major_axis, minor_axis) / max(major_axis, minor_axis)

        if aspect_ratio < 0.77:
            kernel_expand = create_circular_kernel(35)
            opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel_expand)

            contours_disc, _ = cv2.findContours(opened_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours_disc:
                biggest_contour = max(contours_disc, key=cv2.contourArea)
            iterations += 1
        else:
            break

    cv2.ellipse(ellipse_mask, ellipse, 255, -1)
    return ellipse_mask

def calculate_iou(mask1, mask2):

    ground_truth_mask = cv2.imread(mask2, cv2.IMREAD_GRAYSCALE)
    # Compute intersection and union
    intersection = cv2.bitwise_and(mask1, ground_truth_mask)
    union = cv2.bitwise_or(mask1, ground_truth_mask)

    # Calculate IoU
    iou = np.sum(intersection) / np.sum(union)
    return iou


def calcular_cdr(mask_disco, mask_copa):

    filas_disco = np.any(mask_disco == 255, axis=1)
    filas_copa = np.any(mask_copa == 255, axis=1)

    altura_disco = np.sum(filas_disco)
    altura_copa = np.sum(filas_copa)

    cdr = altura_copa / altura_disco if altura_disco > 0 else 0

    return cdr


def plot_mask_perimeters_over_image(image_path, predicted_mask, ground_truth_mask):

    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Error: Unable to load the original image.")
        return

    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    ground_truth = cv2.imread(ground_truth_mask, cv2.IMREAD_GRAYSCALE)

    contours_pred, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_gt, _ = cv2.findContours(ground_truth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay_image = original_image.copy()
    cv2.drawContours(overlay_image, contours_pred, -1, (255, 0, 0), 2)  # Predicted mask in blue
    cv2.drawContours(overlay_image, contours_gt, -1, (0, 255, 0), 2)  # Ground truth mask in green

    plt.figure(figsize=(8, 8))
    plt.title("Rojo: Prediccion  -----  Verde: Ground Truth")
    plt.imshow(overlay_image)
    plt.axis('off')
    plt.show()


def create_circular_kernel(radius):

    diameter = 2 * radius + 1
    kernel = np.zeros((diameter, diameter), dtype=np.uint8)
    center = (radius, radius)
    cv2.circle(kernel, center, radius, 1, -1)

    return kernel


def calcular_mse(valores1, valores2):

    if valores1.shape != valores2.shape:
        raise ValueError("Los arrays deben tener el mismo tamaño.")

    mse = np.mean((valores1 - valores2) ** 2)
    return mse


if __name__ == "__main__":
    # Specify your directories
    input_dir = "refuge_images"
    disc_gt_dir = "disc_images"
    cup_gt_dir = "cup_images"
    cdr_predicted = np.array([])
    cdr_truth = np.array([])
    iou_cup = np.array([])
    iou_disc = np.array([])

    input_images = sorted(os.listdir(input_dir))
    print("-" * 100)
    print(f"{'Image':<25} {'Disc IoU':<10} {'Cup IoU':<10} {'CDR':<10} {'CDR GroundTruth':<10} {'CDR Error':<10}")
    print("-" * 100)

    for image_name in input_images:

        input_path = os.path.join(input_dir, image_name)
        ground_truth_disc = os.path.join(disc_gt_dir, image_name.replace(".png", "_disc.png"))
        ground_truth_cup = os.path.join(cup_gt_dir, image_name.replace(".png", "_cup.png"))

        roi_mask = extract_bounding_box_mask(input_path)
        plot_mask_perimeters_over_image(input_path, roi_mask, ground_truth_disc)

        disc_mask = extract_disc(input_path)
        if disc_mask is not None and os.path.exists(ground_truth_disc):
            disc_iou = calculate_iou(disc_mask, ground_truth_disc)
            plot_mask_perimeters_over_image(input_path, disc_mask, ground_truth_disc)
        else:
            disc_iou = None

        cup_mask = extract_cup(input_path, disc_mask)
        if cup_mask is not None and os.path.exists(ground_truth_cup):
            cup_iou = calculate_iou(cup_mask, ground_truth_cup)
            plot_mask_perimeters_over_image(input_path, cup_mask, ground_truth_cup)
            cdr = calcular_cdr(disc_mask, cup_mask)
            cdr_gt = calcular_cdr(cv2.imread(ground_truth_disc, cv2.IMREAD_GRAYSCALE), cv2.imread(ground_truth_cup, cv2.IMREAD_GRAYSCALE))
        else:
            cup_iou = None

        if disc_iou is not None and cup_iou is not None:
            cdr_error = abs(cdr-cdr_gt)
            print(f"{image_name:<25} {disc_iou:<10.4f} {cup_iou:<10.4f} {cdr:<10.4f} {cdr_gt:<10.4f} {cdr_error:<10.4f}")
            cdr_predicted = np.append(cdr_predicted, cdr)
            cdr_truth = np.append(cdr_truth, cdr_gt)
            iou_cup = np.append(iou_cup, cup_iou)
            iou_disc = np.append(iou_disc, disc_iou)
        elif disc_iou is not None:
            print(f"{image_name:<25} {disc_iou:<10.4f} {'N/A':<10}")
        elif cup_iou is not None:
            print(f"{image_name:<25} {'N/A':<10} {cup_iou:<10.4f}")
        else:
            print(f"{image_name:<25} {'N/A':<10} {'N/A':<10}")

    print("-" * 75)
    print(f"{'Mean Disc IoU':<25} {'Mean Cup IoU':<25} {'CDR Mean Squared Error':<25}")
    print("-" * 75)
    print(f"{np.mean(iou_disc):<25} {np.mean(iou_cup):<25.4f} {calcular_mse(cdr_predicted, cdr_truth):<25.4f}")