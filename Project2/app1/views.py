from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from django.views.decorators.csrf import csrf_exempt
import os
from django.conf import settings
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

@csrf_exempt
def validate(request):
    print("aa")
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        print(image_file)

        file_path = os.path.join(settings.MEDIA_ROOT, image_file.name)
        path = default_storage.save(file_path, ContentFile(image_file.read()))
        img_path = os.path.join(settings.MEDIA_URL, path)
        img_path = img_path[1:]
        print(img_path)
        loaded_model = load_model("app1/validate.h5")

        img = image.load_img(img_path, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        predictions = loaded_model.predict(img_array)
        if predictions[0] < 0.5:
            return JsonResponse({"ML": "YES"})
        print("ML NO")
    return JsonResponse({"ML": "NO"})

# def validate(request):
#     print(request.GET)
#     print(request.POST)
#     print(request.body)
#     loaded_model = load_model("app1/validate.h5")
#     # # img_path = 'app1/uploaded_image.jpg'
#     # img_path = 'app1/image.jpg'
#     # img = image.load_img(img_path, target_size=(256, 256))
#     # img_array = image.img_to_array(img)
#     # img_array = np.expand_dims(img_array, axis=0)
#     # img_array /= 255.0
#     # predictions = loaded_model.predict(img_array)
#     # if predictions[0] < 0.5:
#     #     return JsonResponse({"ML": "YES"})
#     # print("ML NO")
#     return JsonResponse({"ML": "NO"})

