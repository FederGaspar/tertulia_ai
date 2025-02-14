import numpy as np
import cv2
import urllib
import base64
import replicate


class ImageGenerator:

    @staticmethod
    def create_book_image_and_mask(cover, inpainting_color, preservation_color, output_size=(1600, 900)):
        """cover should be encoded in rgba:
            cover = cv2.cvtColor(cv2.imread('image.png'), cv2.COLOR_BGR2RGBA)
            """

        if len(inpainting_color) != 4 or len(preservation_color) != 4:
            raise ValueError("`inapinting_color` and `preservation_color` should RGBA colors")

        # pasar a un alto de 648
        front = cv2.resize(cover, (cover.shape[1] * 648 // cover.shape[0], 648))
        back = cv2.GaussianBlur(front, (125, 125), 100)
        a, b = output_size
        mask = np.zeros((b, a, 4), dtype='uint8')
        mask[:, :] = inpainting_color
        cover_shadow = np.zeros((front.shape[0], front.shape[1], 4), dtype='uint8')
        cover_shadow[:, :] = preservation_color

        shift = [1000 - cover.shape[1] // 2, 150]

        sheets = np.full((front.shape[0], 100, 4), 255, dtype='uint8')
        sheets[:] = [240, 240, 240, 255]
        sheets[:, ::4] = [220, 220, 220, 255]
        sheets[0] = [0, 0, 0, 255]
        sheets[-1] = [0, 0, 0, 255]

        sheets_shadow = np.zeros_like(sheets)
        sheets_shadow[::] = preservation_color

        h, w = back.shape[:2]
        p0 = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype='float32')
        p1 = np.array([[50, 50], [50, h - 50], [w, h - 10], [w, 10]], dtype='float32')
        p1 += shift
        M = cv2.getPerspectiveTransform(p0, p1)
        result_image = cv2.warpPerspective(back, M, output_size)
        result_mask = cv2.warpPerspective(cover_shadow, M, output_size, mask, borderMode=cv2.BORDER_TRANSPARENT)

        hh, ww = sheets.shape[:2]
        p0 = np.array([[0, 0], [0, hh], [ww, hh], [ww, 0]], dtype='float32')
        p1 = np.array([[w - 80, 10], [w - 80, h - 10], [w - 10, h - 20], [w - 10, 20]], dtype='float32')
        p1 += shift
        M = cv2.getPerspectiveTransform(p0, p1)
        result_image = cv2.warpPerspective(sheets, M, output_size, result_image, borderMode=cv2.BORDER_TRANSPARENT)
        result_mask = cv2.warpPerspective(sheets_shadow, M, output_size, result_mask, borderMode=cv2.BORDER_TRANSPARENT)

        p0 = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype='float32')
        p1 = np.array([[-20, 40], [-20, h - 40], [w - 70, h], [w - 70, 0]], dtype='float32')
        p1 += shift
        M = cv2.getPerspectiveTransform(p0, p1)
        result_image = cv2.warpPerspective(front, M, output_size, result_image, borderMode=cv2.BORDER_TRANSPARENT)
        result_mask = cv2.warpPerspective(cover_shadow, M, output_size, result_mask, borderMode=cv2.BORDER_TRANSPARENT)

        _, image_arr = cv2.imencode('.png', result_image)  # im_arr: image in Numpy one-dim array format.
        image_b64 = base64.b64encode(image_arr)

        _, mask_arr = cv2.imencode('.png', result_mask)  # im_arr: image in Numpy one-dim array format.
        mask_b64 = base64.b64encode(mask_arr)

        image_b64 = image_b64.decode("utf-8")
        mask_b64 = mask_b64.decode("utf-8")

        return image_b64, mask_b64

    @staticmethod
    def ideogram_v2_inpainting(image, mask, prompt):
        image_data = f"data:application/octet-stream;base64,{image}"
        mask_data = f"data:application/octet-stream;base64,{mask}"

        response = replicate.run(
            "ideogram-ai/ideogram-v2-turbo",
            input={"prompt": prompt,
                   "image": image_data,
                   "mask": mask_data,
                   "style_type": "Realistic"}
        )
        return base64.b64encode(response.read())

    def generate_image(self, ean, prompt):
        r = urllib.request.urlopen(f'https://d16057n354qyo4.cloudfront.net/{ean}.jpg')
        a = np.asarray(bytearray(r.read()), dtype=np.uint8)
        cover = cv2.cvtColor(cv2.imdecode(a, -1), cv2.COLOR_BGR2BGRA)
        book_image, mask_image = self.create_book_image_and_mask(cover, [0,0,0,255], [255,255,255,255])
        return {"generated_image": self.ideogram_v2_inpainting(book_image, mask_image, prompt),
                "book_image": book_image,
                "mask_image": mask_image}