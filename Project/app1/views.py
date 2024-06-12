# myapp/views.py
from django.http import JsonResponse
from django.views import View
from . import model_service
import numpy as np
from django.views.decorators.csrf import csrf_exempt
import os
from django.conf import settings
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

def PredictView(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        print(image_file)

        file_path = os.path.join(settings.MEDIA_ROOT, image_file.name)
        path = default_storage.save(file_path, ContentFile(image_file.read()))
        img_path = os.path.join(settings.MEDIA_URL, path)
        img_path = img_path[1:]
        print(img_path)

        output = model_service.predict(img_path)

        response = {
            'prediction': output.tolist()
            # 'prediction': output
        }
        return JsonResponse(response)
