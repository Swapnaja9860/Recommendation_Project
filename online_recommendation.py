#from flask import Flask, render_template,request,redirect,url_for,flash,
import tkinter
from tkinter import messagebox
from flask import Flask, redirect, url_for, request, render_template , session
import csv
app = Flask(__name__)
app.secret_key = "any random string"


from one import one
from second import second
from user import user
from item import item

app = Flask(__name__)
app.secret_key = "any random string"
app.register_blueprint(one , url_prefix="/admin")
app.register_blueprint(second , url_prefix="/algo")
app.register_blueprint(user, url_prefix="/algo1")
app.register_blueprint(item, url_prefix="/algo2")



@app.route('/front')
def front():
  return render_template("trial_front.html")

@app.route('/algo/<userid>')
def algo(userid):
  userid = int(userid)
  return render_template("algo.html",userid=userid)

@app.route('/aboutus')
def aboutus():
  return render_template("about_us.html")

@app.route('/abcde/<plce>/<image>')
def place_trial(plce,image):
  data = []
  with open('.data/describe_final_final.csv',encoding="utf8",newline="" ) as csv_file:
   reader = csv.reader(csv_file , delimiter='"' , quoting=csv.QUOTE_NONE)
   for row in reader:
    data.append(row)
  col0 = [x[0] for x in data]
  col1 = [x[1] for x in data]
  places = []
  describe = "No info"
  if plce in col0 :     
    for k in range(0,len(col0)):
      if plce == col0[k] :
        describe = col1[k] 
        '''places.append({
          "Place" : col0[k],
          "description" : col1[k],
          "image" : image
        })'''
  return render_template("view.html", describe= describe , plce= plce ,image = image)
            
@app.route('/abcd')
def place():
  data = []
  with open('.data/trial1.csv',newline="" ) as csv_file:
   reader = csv.reader(csv_file , delimiter='"' , quoting=csv.QUOTE_NONE)
   for row in reader:
    data.append(row)
  col0 = [x[0] for x in data]
  col1 = [x[1] for x in data]
  col2 = [x[2] for x in data]
  col3 = [x[3] for x in data]
  col4 = [x[4] for x in data]
  places = []
        
  for k in range(0,len(col2)):
    places.append({
      "PlaceID" : col0[k],
      "category" : col1[k],
      "Place" : col2[k],
      "image" : col3[k], 
      "reviews" : col4[k]
    })
  return render_template("trial2.html", places= places)

@app.route('/login', methods=['POST', 'GET'])
def login():
  if request.method == 'POST':
      login = False
      while login == False :
        data = []
        with open(".data/user.csv") as csv_file:
          reader = csv.reader(csv_file)
          for row in reader:
            data.append(row)    
        users = dict(request.form)
        email = users["email"]
        password = users["password"]
        col0 = [x[2] for x in data]
        col1 = [x[3] for x in data]
        col2 = [x[0] for x in data]
        print(col0)
        t = 1
        abcd = len(col0)
        if email in col0 :
          t = t+1
          for k in range(0,len(col0)):
            if email == col0[k]:
              if password == col1[k]:
                login = True 
                userid = col2[k]
                session['userid'] = userid
                session['email'] = email
                #return redirect(url_for('index1'))
                return redirect(url_for('popular_place' , userid = userid));
              else :
                return redirect("reenter password");
              #return "login successful"
        else :
          messagebox.showinfo("title","Please Enter Correct email ID")
          return render_template('login.html');
          #return render_template('trial.html',email = email,password = password,col0=col0,col1=col1,t = t,data = data,abcd = abcd)    
        #return redirect(url_for('places', userid=user))
    #else:
  return render_template('login.html')

@app.route('/reco_places/<userid>')
def reco_places(userid):
    User = int(userid)
    return render_template('reco_places.html', userid = User)     

@app.route('/')
def index1():
   if 'email' in session:
      email = session['email']
      return 'Logged in as ' + email + '<br>' + "<b><a href = '/logout'>click here to log out</a></b>"
   return "You are not logged in <br><a href = '/login'>" + "click here to log in</a>"

@app.route('/logout')
def logout():
   # remove the username from the session if it is there
   session.pop('email', None)
   return redirect(url_for('index1'))

@app.route('/view/<placeid>')
def view(placeid):
    placeid = int(placeid)
    return render_template('view.html', placeid = placeid) 

@app.route('/algorithm/<userid>')
def algorithm(userid):
    User = int(userid)
    return render_template('algorithm.html', userid = User)

@app.route('/popular_place/<userid>')
def popular_place(userid):
    userid = int(userid)
    return render_template('popular_place.html',userid = userid)

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/abcde')
def abcde():
    return render_template('login.html')

@app.route('/rate/<place>')
def rating(place):
    return render_template('rating.html', Place = place)

@app.route('/trial1')
def fun():
  with open('.data/category.csv') as csv_file:
    data = csv.reader(csv_file, delimiter=',')
    first_line = True
    places = []
    for row in data:
      if not first_line:
        places.append({
          "Place": row[1],
          "image" : row[2] 
        })
      else:
        first_line = False
  return render_template("category.html", places=places)

@app.route('/trial/<category>')
def trial(category):
  #category = "Nature & Parks" ;
  data = []
  with open(".data/place_dataset.csv",mode="r") as fi:
    reader = csv.reader(fi,delimiter=",")
    for row in reader:
        data.append(row)    
    col0 = [x[0] for x in data]
    col1 = [x[1] for x in data]
    col2 = [x[2] for x in data]
    col3 = [x[3] for x in data]
    
    places=[]
    if category in col1:
      for k in range(0,len(col1)):
        if category == col1[k]:
           places.append({
                            "PlaceID" : col0[k],
                            "category" : col1[k],
                            "Place" : col2[k],
                            "image" : col3[k] 
                        })
      return render_template("category_places.html" , places = places , category=category)
    return render_template("popular_place.html")


@app.route("/search" , methods=['POST', 'GET'])
def search():
  if request.method == 'POST' :
    userdata = dict(request.form)
    search = userdata.get("search")
    data = []
    with open(".data/place_dataset.csv",mode="r") as fi:
      reader = csv.reader(fi,delimiter=",")
      for row in reader:
          data.append(row)    
      col0 = [x[0] for x in data]
      col1 = [x[1] for x in data]
      col2 = [x[2] for x in data]
      col3 = [x[3] for x in data]
    
      places=[]
      if search in col2:
        for k in range(0,len(col2)):
          if search == col2[k]:
             places.append({
                              "PlaceID" : col0[k],
                              "category" : col1[k],
                              "Place" : col2[k],
                              "image" : col3[k] 
                          })
        return render_template("trial3.html" , places = places)
  return render_template("popular_place.html")

@app.route("/data/<Plce>" , methods=['POST', 'GET'])
def data(Plce):
  if request.method == 'POST' :
    places = session['places']
    for place in places:
      if place['Place'] == Plce :
        UserID = session.get("userid")
        PlaceID = place['PlaceID']
        userdata = dict(request.form)
        rate = userdata.get("rating")
        with open('.data/Main_dataset.csv',mode='a',newline='') as csv_file:
          data = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
          data.writerow([UserID,PlaceID,rate])
          csv_file.close();
  return render_template('popular_place.html')


@app.route("/submit", methods=["GET", "POST"])
def submit():
  if request.method == "GET":
    return redirect(url_for('signup'))
  elif request.method == "POST":
    with open('.data/user.csv', mode='r') as f:
     reader = csv.reader(f,delimiter = ",")
     data = list(reader)
    row_count = len(data)
    row = data[row_count-1]
    userid = row[0]
    #f.close()
    userdata = dict(request.form)
    userid = int(userid)
    userid = userid + 1
    session['userid'] = userid
    fname = userdata["fname"]
    email = userdata["email"]
    password = userdata["password"]
    gender = userdata["gender"]
    with open('.data/user.csv', mode='a', newline='') as csv_file:
      data = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      data.writerow([userid,fname, email,password,gender])
      csv_file.close();
  return redirect(url_for('popular_place' , userid = userid));

@app.route('/item_reco')
def item_reco() :
  with open('.data/explore_final.csv') as csv_file:
    data = csv.reader(csv_file, delimiter=',')
    first_line = True
    places = []
    for row in data:
      if not first_line:
        places.append({
          "PlaceID": row[0],
          "Place": row[1],
          "image" : row[2] 
        })
      else:
        first_line = False
  return render_template("item.html", places=places)

@app.route("/s")
def index():
  with open('.data/explore.csv') as csv_file:
    data = csv.reader(csv_file, delimiter=',')
    first_line = True
    places = []
    for row in data:
      if not first_line:
        places.append({
          "Place": row[0],
          "image" : row[1] 
        })
      else:
        first_line = False
  return render_template("explore.html", places=places)

if __name__ == '__main__':
    app.debug = True
    app.run()
    app.run(debug=True)
