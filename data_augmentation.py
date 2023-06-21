import cv2
import os
import numpy as np
from skimage.util import random_noise



def data_augmentation(image_path, folder_path, number_of_variations=16):
    # Carregar a imagem original

    # Obter o nome do arquivo sem a extensão
    filename = os.path.splitext(os.path.basename(image_path))[0]

    # percorrer o numero de variações
    for i in range(number_of_variations):
        image = cv2.imread(image_path)

        # chance de 1 em 6 de aplicar gaussian
        salt_pepper_chance = np.random.randint(0, 6)
        if salt_pepper_chance == 1:
            gaussian_noise = random_noise(image, mode='gaussian')

            # passar a imagem para o proximo passo
            image = np.array(255 * gaussian_noise, dtype='uint8')


        # chance de 1 em 8 de colocar um retangulo preto na imagem
        black_rectangle_chance = np.random.randint(0, 8)
        if black_rectangle_chance == 1:
            # Retangulo preto
            height, width = image.shape[:2]
            start_row, start_col = int(height * .25), int(width * .25)
            end_row, end_col = int(height * .75), int(width * .75)
            cv2.rectangle(image, (start_col, start_row), (end_col, end_row), (0, 0, 0), -1)
            # passar a imagem para o proximo passo
            image = image


        # Espelhar a imagem
        # chance de 50% de espelhar horizontalmente
        flip_chance = np.random.randint(0, 2)
        if flip_chance == 1:
            flipped_image = cv2.flip(image, 1)

            # passar a imagem para o proximo passo
            image = flipped_image


        # Rotacionar a imagem de -90 a 90 graus
        rotation_angle = np.random.randint(-60, 60)
        height, width = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), rotation_angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

        # passar a imagem para o proximo passo

        image = rotated_image
        # chance de 1 em 5 de aplicar blur
        blur_chance = np.random.randint(0, 5)
        if blur_chance == 1:
            # Blur
            blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
            # passar a imagem para o proximo passo
            image = blurred_image

        # chance de 1 em 5 de aplicar sharpen
        sharpen_chance = np.random.randint(0, 8)
        if sharpen_chance == 1:
            # Sharpen
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened_image = cv2.filter2D(image, -1, kernel)
            # passar a imagem para o proximo passo
            image = sharpened_image

        # pequeno intervalo para mudar o brilho e contraste
        alpha = np.random.uniform(0.5, 1.2)
        beta = np.random.randint(-5, 5)

        # Mudar o brilho e contraste da imagem
        brightness_contrast_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # passar a imagem para o proximo passo
        image = brightness_contrast_image

        # pequeno intervalo para mudar a saturação
        saturation = np.random.uniform(0.5, 1.1)

        # Mudar a saturação da imagem
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 1] = hsv_image[:, :, 1] * saturation
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


        # chance de 1 em 8 de aplicar central crop
        central_crop_chance = np.random.randint(0, 10)
        if central_crop_chance == 1:
            # Central crop
            height, width = image.shape[:2]
            start_row, start_col = int(height * .25), int(width * .25)
            end_row, end_col = int(height * .75), int(width * .75)
            cropped_image = image[start_row:end_row, start_col:end_col]
            # passar a imagem para o proximo passo
            image = cropped_image
            


        # aplicar ruido gaussiano
        # chance de 1 em 8 de aplicar ruido gaussiano
   


        # salva a imagem
        cv2.imwrite(f'./train_images/{folder_path}/{filename}_{i}.jpg', image)
        
        


  
    

# Criar só as 5 primeiras variações
#for folder in os.listdir('./train_images'):    
#    for image in os.listdir(f'./train_images/{folder}')[0:5]:
#        image_path = f'./train_images/{folder}/{image}'
#        data_augmentation(image_path, folder)


# Salvar variações
#for folder in os.listdir('./train_images'):
#    for image in os.listdir(f'./train_images/{folder}'):
#        image_path = f'./train_images/{folder}/{image}'
#        data_augmentation(image_path, folder)

# Remover variações
for folder in os.listdir('./a_train_images'):
     for image in os.listdir(f'./a_train_images/{folder}'):
# se a imagem tiver dois _ no nome, ela é uma variação errada
        if image.count('_') == 2:
            os.remove(f'./a_train_images/{folder}/{image}')





