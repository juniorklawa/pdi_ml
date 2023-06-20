import cv2
import os
import numpy as np


def data_augmentation(image_path, folder_path):
    # Carregar a imagem original
    image = cv2.imread(image_path)

    # Obter o nome do arquivo sem a extensão
    filename = os.path.splitext(os.path.basename(image_path))[0]

    # Espelhar a imagem
    flipped_image = cv2.flip(image, 1)
    cv2.imwrite(f'./train_images/{folder_path}/{filename}_flipped.jpg', flipped_image)

    # Rotacionar a imagem
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(f'./train_images/{folder_path}/{filename}_rotated.jpg', rotated_image)

    # Blur na imagem
    blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
    cv2.imwrite(f'./train_images/{folder_path}/{filename}_blurred.jpg', blurred_image)

    # Mais blur
    more_blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
    cv2.imwrite(
        f'./train_images/{folder_path}/{filename}_more_blurred.jpg', more_blurred_image)

    # Sharpen
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    cv2.imwrite(f'./train_images/{folder_path}/{filename}_sharpened.jpg', sharpened_image)

    # Aumentar o brilho da imagem
    brightness_image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    cv2.imwrite(f'./train_images/{folder_path}/{filename}_brightness.jpg', brightness_image)

    # Diminuir o brilho da imagem
    darkness_image = cv2.convertScaleAbs(image, alpha=0.5, beta=0)
    cv2.imwrite(f'./train_images/{folder_path}/{filename}_darkness.jpg', darkness_image)

    # Aumentar o contraste da imagem
    contrast_image = cv2.convertScaleAbs(image, alpha=1.0, beta=50)
    cv2.imwrite(f'./train_images/{folder_path}/{filename}_contrast.jpg', contrast_image)

    # Change hue 180, 90, 45
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + 120) % 180
    hue_120_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(f'./train_images/{folder_path}/{filename}_hue_120.jpg', hue_120_image)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + 90) % 180
    hue_90_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(f'./train_images/{folder_path}/{filename}_hue_90.jpg', hue_90_image)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + 45) % 180
    hue_45_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(f'./train_images/{folder_path}/{filename}_hue_45.jpg', hue_45_image)
    

    # Diminuir o contraste da imagem
    low_contrast_image = cv2.convertScaleAbs(image, alpha=1.0, beta=-50)
    cv2.imwrite(
        f'./train_images/{folder_path}/{filename}_low_contrast.jpg', low_contrast_image)

    # Dilatar a imagem
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_image = cv2.dilate(image, kernel)
    cv2.imwrite(f'./train_images/{folder_path}/{filename}_dilated.jpg', dilated_image)

    # Erodir a imagem
    eroded_image = cv2.erode(image, kernel)
    cv2.imwrite(f'./train_images/{folder_path}/{filename}_eroded.jpg', eroded_image)

    # 50% JPEG QUALITYT
    cv2.imwrite(f'./train_images/{folder_path}/{filename}_50.jpg', image, [
                cv2.IMWRITE_JPEG_QUALITY, 50])
    
    # 25% JPEG QUALITYT
    cv2.imwrite(f'./train_images/{folder_path}/{filename}_25.jpg', image, [
                cv2.IMWRITE_JPEG_QUALITY, 25])
    # 10% JPEG QUALITYT
    cv2.imwrite(f'./train_images/{folder_path}/{filename}_10.jpg', image, [
                cv2.IMWRITE_JPEG_QUALITY, 10])
    
    # Gray scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'./train_images/{folder_path}/{filename}_gray.jpg', gray_image)

    # Saturation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = (hsv_image[:, :, 1] + 180) % 180
    saturation_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(f'./train_images/{folder_path}/{filename}_saturation.jpg', saturation_image)

    #Central crop
    height, width = image.shape[:2]
    start_row, start_col = int(height * .25), int(width * .25)
    end_row, end_col = int(height * .75), int(width * .75)
    cropped_image = image[start_row:end_row, start_col:end_col]
    cv2.imwrite(f'./train_images/{folder_path}/{filename}_cropped.jpg', cropped_image)

  
    

# Percorrer todas as imagens da pasta Lotus, Lilly, Orchid, Sunflower e Tulip
# e aplicar a função data_augmentatio, mas apenas as 5 primeiras imagens de cada pasta
# for folder in os.listdir('./train_images'):    
#     for image in os.listdir(f'./train_images/{folder}')[0:5]:
#         image_path = f'./train_images/{folder}/{image}'
#         data_augmentation(image_path, folder)


# Percorrer todas as imagens da pasta Lotus, Lilly, Orchid, Sunflower e Tulip
# e aplicar a função data_augmentatio, salvando no memso diretório 
for folder in os.listdir('./train_images'):
    for image in os.listdir(f'./train_images/{folder}'):
        image_path = f'./train_images/{folder}/{image}'
        data_augmentation(image_path, folder)

# now remove all augmented images from the train folder (that contains "_" character)
# for folder in os.listdir('./train_images'):
#     for image in os.listdir(f'./train_images/{folder}'):
#         if "_" in image:
#             os.remove(f'./train_images/{folder}/{image}')




