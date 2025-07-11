import os
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math


class ImageAttacks:
    def __init__(self, img=None, img_path=None):
        if img_path:
            self.orig_img = cv2.imread(img_path)
            if self.orig_img is None:
                raise ValueError(f"无法读取图像: {img_path}")
            self.orig_img = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2RGB)
        elif img is not None:
            self.orig_img = img  # 直接使用 numpy 数组
        else:
            raise ValueError("必须提供 img 或 img_path")

    def color_saturation(self, severity):

        saturation_values = {
            0: "No",
            1: 0.4,
            2: 0.3,
            3: 0.2,
            4: 0.1,
            5: 0.0
        }

        saturation = saturation_values[severity]
        if saturation == "No":
            return self.orig_img.copy()

        img_hsv = cv2.cvtColor(self.orig_img, cv2.COLOR_RGB2HSV).astype(np.float32)
        # 调整饱和度通道
        img_hsv[:, :, 1] = img_hsv[:, :, 1] * saturation
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
        img_output = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return img_output

    def color_contrast(self, severity):

        contrast_values = {
            0: "No",
            1: 0.85,
            2: 0.725,
            3: 0.6,
            4: 0.475,
            5: 0.35
        }

        contrast = contrast_values[severity]
        if contrast == "No":
            return self.orig_img.copy()

        img = self.orig_img.copy().astype(np.float32)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        img = contrast * (img - mean) + mean
        img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def blockwise(self, severity):

        block_counts = {
            0: "No",
            1: 16,
            2: 24,
            3: 32,
            4: 48,
            5: 64
        }

        blocks = block_counts[severity]
        if blocks == "No":
            return self.orig_img.copy()

        img = self.orig_img.copy()
        h, w, c = img.shape

        # 创建一个随机的块尺寸，基于图像大小和块数
        block_size = 16

        for _ in range(blocks):
            # 随机选择块的位置
            block_x = np.random.randint(0, w - block_size)
            block_y = np.random.randint(0, h - block_size)

            # 将块像素置为0（显示为黑色块）
            # 如果需要灰色块，可以设置为灰色值，例如[128, 128, 128]
            img[block_y:block_y + block_size, block_x:block_x + block_size] = 0

        return img

    def gaussian_noise(self, severity):
        """添加高斯噪声

        Args:
            severity: 严重程度 (1-5)

        Returns:
            处理后的图像
        """
        image = self.orig_img.copy().astype(np.float32)
        # 增强噪声强度，调整 severities 列表
        severities = [0.02, 0.04, 0.06, 0.08, 0.1]  # 显著增加噪声强度
        noise_level = severities[severity - 1]
        # 放大噪声强度，引入额外的放大系数
        noise = np.random.normal(0, noise_level * 255, image.shape)  # 放大系数 1.5
        noisy_image = image + noise
        # 确保像素值在有效范围内
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image.astype(np.uint8)

    def gaussian_blur(self, severity):
        """应用高斯模糊

        Args:
            severity: 严重程度 (1-5)

        Returns:
            处理后的图像
        """
        blur_kernels = {
            0: "No",
            1: (3, 3),
            2: (5, 5),
            3: (7, 7),
            4: (9, 9),
            5: (13, 13)
        }

        kernel_size = blur_kernels[severity]
        if kernel_size == "No":
            return self.orig_img.copy()

        blurred_img = cv2.GaussianBlur(self.orig_img, kernel_size, 0)

        return blurred_img

    def jpeg_compression(self, severity):
        """应用JPEG压缩

        Args:
            severity: 严重程度 (1-5)

        Returns:
            处理后的图像
        """
        quality_values = {
            0: "No",
            1: 90,
            2: 70,
            3: 50,
            4: 30,
            5: 20
        }

        quality = quality_values[severity]
        if quality == "No":
            return self.orig_img.copy()

        img = Image.fromarray(self.orig_img)
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=quality)
        output.seek(0)
        compressed_img = np.array(Image.open(output))

        return compressed_img

    def rotate(self, severity):
        """旋转图像

        Args:
            severity: 严重程度 (1-5)

        Returns:
            处理后的图像
        """
        angle_values = {
            0: "No",
            1: 30,
            2: 60,
            3: 90,
            4: 120,
            5: 150
        }

        angle = angle_values[severity]
        if angle == "No":
            return self.orig_img.copy()

        h, w = self.orig_img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(self.orig_img, rotation_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

        return rotated_img

    def affine_transformation(self, severity):
        """应用仿射变换（旋转10度，平移10像素，然后按不同程度缩放）

        Args:
            severity: 严重程度 (1-5)

        Returns:
            处理后的图像
        """
        scale_values = {
            0: "No",
            1: 1.2,
            2: 1.1,
            3: 1.0,
            4: 0.9,
            5: 0.8
        }

        scale = scale_values[severity]
        if scale == "No":
            return self.orig_img.copy()

        h, w = self.orig_img.shape[:2]
        center = (w // 2, h // 2)

        # 步骤1: 创建旋转10度的旋转矩阵
        angle = 10
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 步骤2: 添加10像素的平移
        translation_x = 10
        translation_y = 10
        rotation_matrix[0, 2] += translation_x
        rotation_matrix[1, 2] += translation_y

        # 步骤3: 应用缩放比例（根据严重程度）
        # 注意：在OpenCV中，这需要重新创建变换矩阵
        scale_matrix = cv2.getRotationMatrix2D(center, 0, scale)

        # 组合变换：先旋转和平移，然后缩放
        # 先应用旋转和平移
        rotated_img = cv2.warpAffine(self.orig_img, rotation_matrix, (w, h),
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 再应用缩放
        final_img = cv2.warpAffine(rotated_img, scale_matrix, (w, h),
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        return final_img

    def apply_all_attacks(self, output_dir="output"):
        """应用所有攻击并保存结果

        Args:
            output_dir: 输出目录
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        attacks = [
            ('CS', self.color_saturation),
            ('CC', self.color_contrast),
            ('BW', self.blockwise),
            ('GN', self.gaussian_noise),
            ('GB', self.gaussian_blur),
            ('JPEG', self.jpeg_compression),
            ('RO', self.rotate),
            ('AF', self.affine_transformation)
        ]

        # 创建一个大的画布来展示所有结果
        fig, axs = plt.subplots(6, 9, figsize=(20, 15))

        # 设置第一行为标题
        axs[0, 0].text(0.5, 0.5, 'Original', ha='center', va='center', fontsize=12)
        for i, (attack_name, _) in enumerate(attacks, 1):
            axs[0, i].text(0.5, 0.5, attack_name, ha='center', va='center', fontsize=12)

        # 原始图像
        axs[1, 0].imshow(self.orig_img)
        axs[1, 0].set_title('Severity 0')
        axs[1, 0].axis('off')

        # 应用每种攻击的每个严重程度
        for severity in range(1, 6):
            axs[severity, 0].text(0.5, 0.5, f'Severity {severity}', ha='center', va='center', fontsize=12)
            axs[severity, 0].axis('off')

            for i, (attack_name, attack_func) in enumerate(attacks, 1):
                # 应用攻击
                attacked_img = attack_func(severity)

                # 显示结果
                axs[severity, i].imshow(attacked_img)
                axs[severity, i].set_title(f'{attack_name} - S{severity}')
                axs[severity, i].axis('off')

                # 保存单独的图像
                plt.imsave(os.path.join(output_dir, f'{attack_name}_S{severity}.png'), attacked_img)

        # 调整布局并保存总览图
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'all_attacks_overview.png'), dpi=300)
        plt.close()

        print(f"所有处理后的图像已保存到 {output_dir} 目录")


# 使用示例
if __name__ == "__main__":
    import io

    # 替换为您的输入图像路径
    input_image = "modules/frame_0.jpg"

    # 创建攻击对象
    attacks = ImageAttacks(input_image)

    # 应用所有攻击并保存结果
    attacks.apply_all_attacks(output_dir="robustness_results")