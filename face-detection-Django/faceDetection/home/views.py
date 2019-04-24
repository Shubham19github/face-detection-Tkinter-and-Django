from django.shortcuts import render
from django.http import HttpResponse

# to stop the webcam and release the resourc
from live_feed.views import toDelete_lf

# Create your views here.

video = False

def home_page_view(request, *args, **kwargs):
    global video

    # when start camera button is pressed
    if(request.POST.get('mybtn')):
        video = True

    # when stop camera button is pressed
    elif(request.POST.get('closeBtn')):
        video = False
        # calling function to release the webcam
        toDelete_lf()
        
    homepage_context = {
        "video": video,
    }

    return render(request, "home.html", homepage_context)
