from flask import Blueprint, Flask, redirect, url_for, request, render_template

distance = Blueprint("distance",__name__, static_folder="static",template_folder="templates")

@distance.route('/view/<place>')
def view(place):
    address = abcd(place)    
    return render_template('trial1.html', address = address)
#!/usr/bin/env python
# coding: utf-8

from __main__ import *
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
geolocator = Nominatim(scheme='http', timeout=3)

def abcd(place):
    location = geolocator.geocode(place) 
    x= float(location.latitude)
    y= float(location.longitude)
    location = geolocator.reverse([x,y]) 
    #details = location.raw
    add = location.address
    return add 
       