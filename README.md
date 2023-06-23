
::page{title="Working with Routes in Flask "}


<img src="/images/SN_web_lightmode.png" width=200/>

# Working with Routes
##

Estimated time needed: 45 minutes

## 
Welcome to the Flask lab. In this section, you will work with routes and HTTP requests in Flask. You will learn how to create routes to process requests at specific URLs, render templates and handle parameters and arguments sent to the URLs, Ensure that you have basic knowledge of the Flask. Feel free to pause the lab and review all the concept of the flask if needed.

::page{title=""}

## Learning Objectives
After completing this lab, you will be able to:

- Write routes to process requests to the Flask server at specific URLs
- Render Templates
- Handle parameters and arguments sent to the URLs

# About Skills Network Cloud IDE
Skills Network Cloud IDE (based on Theia and Docker) provides an environment for hands on labs for course and project related labs. Theia is an open source IDE (Integrated Development Environment) that runs on desktop or the cloud. To complete this lab, you will use the Cloud IDE based on Theia.

## Important Notice about this lab environment
Please be aware that sessions do not persist for this lab environment. Every time you connect to this lab, a new environment is created for you. Any data saved in earlier sessions will be lost. Plan to complete these labs in a single session to avoid losing your data.

# Set Up the Lab Environment
There are some required prerequisite preparations before you start the lab.
## Open a Terminal
Open a terminal window using the menu in the editor: Terminal > New Terminal.

In the terminal, if you are not in the /home/project folder, change to your project folder now.

```bash
cd /home/project
```
# Create the lab directory
You should have a lab directory of the lab.Create it now.
```bash
mkdir lab
```
# Change to the lab directory:
```bash
cd lab
```
Create a server.py inside the lab directory.
```bash
touch server.py
```
Open the server.py file in the editor.

Step 1: Import Flask and Initialize the App
- In this step, you need to import the necessary modules (`Flask`, `render_template`, and `request`) and create a Flask application instance.

Double-check that your work matches the solution below.
<details><summary><i>👉 Click here for the answer.</i></summary>
  
```python
  
from flask import Flask, render_template, request

app = Flask(__name__)

```
</details>

Step 2: Create the Home Route
- Define a route for the home page ("/") using the `@app.route` decorator.
- Inside the route function, return a response by rendering a template called "home.html" and passing a variable `name` with the value 'John'. Use the `render_template` function from Flask.

Double-check that your work matches the solution below.
<details><summary><i>👉 Click here for the answer.</i></summary>
  
```python
  
@app.route('/')
def home():
    name = 'John'
    return render_template('home.html', name=name)
```
</details>

Create an HTML file called "home.html" in your templates directory.
Use the appropriate HTML tags to structure the page.
Display a heading that says "Welcome to the Home Page!"
Use the {{ name }} variable to display the value passed from the route function.

```
<!DOCTYPE html>
<html>
<head>
    <title>My Flask App</title>
    <style>
        html, body {
            height: 100%;
            margin: 0;
            padding: 0;
        }

        body {
            display: flex;
            flex-direction: column;
            font-family: Arial, sans-serif;
            background-color: #f2f2f2;
        }

        header {
            background-color: #333333;
            padding: 20px;
            color: #ffffff;
            text-align: center;
        }

        main {
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }

        footer {
            background-color: #333333;
            padding: 10px;
            color: #ffffff;
            text-align: center;
            font-size: 12px;
        }

        h1 {
            color: #333333;
            text-align: center;
            margin-top: 50px;
        }

        p {
            color: #666666;
            text-align: center;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <header>
        <h1>My Flask App</h1>
    </header>

    <main>
        <h1>Hello, {{ name }}!</h1>
        <p>Welcome to my Flask app.</p>
    </main>

    <footer>
        &copy; 2023 My Flask App. All rights reserved.
    </footer>
</body>
</html>

```


Step 3: Create the About Route
- Define a route for the about page ("/about") using the `@app.route` decorator.
- Inside the route function, return a string response that represents an HTML page. You can use a multi-line string or create a separate HTML file and return its contents.

Double-check that your work matches the solution below.
<details><summary><i>👉 Click here for the answer.</i></summary>
  
```python
@app.route('/about')
def about():
    return '''
    <html>
            <head>
                <title>About</title>
                <style>
                    .about-page {
                        font-family: Arial, sans-serif;
                        font-size: 18px;
                        color: #333333;
                        text-align: center;
                        padding: 50px;
                        background-color: #f2f2f2;
                    }
                </style>
            </head>
            <body>
                <div class="about-page">
                    This is the about page.
                </div>
            </body>
        </html>
    '''
```
</details>
Step 4: Create the Contact Route
- Define a route for the contact page ("/contact") using the `@app.route` decorator.
- Inside the route function, return a response by rendering a template called "contact.html" using the `render_template` function.

Double-check that your work matches the solution below.
<details><summary><i>👉 Click here for the answer.</i></summary>
  
```python
@app.route('/contact')
def contact():
    return render_template('contact.html')
```
</deatils>

Create an HTML file called "contact.html" in your templates directory.
Use the appropriate HTML tags to structure the page.
Add a contact form with fields for name, email, and message.
Include a submit button for the form.

```
<!DOCTYPE html>
<html>
<head>
    <title>My Flask App</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f2f2f2;
        }

        h1 {
            color: #333333;
            text-align: center;
            margin-top: 50px;
        }

        p {
            color: #666666;
            text-align: center;
            margin-top: 20px;
        }

        form {
            max-width: 400px;
            margin: 0 auto;
            background-color: #ffffff;
            padding: 20px;
            border-radius: 5px;
        }

        input[type="text"],
        textarea {
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid #cccccc;
            border-radius: 4px;
            box-sizing: border-box;
        }

        input[type="submit"] {
            background-color: #4caf50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }

        input[type="submit"]:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <h1>Hello!</h1>
    <p>Contact Us.</p>

    <form>
        <input type="text" placeholder="Name" name="name">
        <input type="text" placeholder="Email" name="email">
        <textarea placeholder="Message" name="message"></textarea>
        <input type="submit" value="Submit">
    </form>
</body>
</html>

```


Step 5: Create the User Profile Route
- Define a route with a dynamic parameter for the username ("/users/<username>") using the `@app.route` decorator.
- Inside the route function, retrieve the dynamic parameter using a parameter in the function definition (e.g., `def user_profile(username):`).
- Return a response by rendering a template called "profile.html" and pass the `username` variable to the template.

Double-check that your work matches the solution below.
<details><summary><i>👉 Click here for the answer.</i></summary>
  
```python
@app.route('/users/<username>')
def user_profile(username):
    return render_template('profile.html', username=username)
```
</details>
  
Create an HTML file called "profile.html" in your templates directory.
Use the appropriate HTML tags to structure the page.
Display a heading that says "User Profile: {{ username }}"
Include additional information or sections as desired to showcase the user's profile.

```
<!DOCTYPE html>
<html>
<head>
    <title>User Profile</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f2f2f2;
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }

        h1 {
            color: #333333;
            text-align: center;
            margin-top: 50px;
        }

        h2 {
            color: #666666;
            text-align: center;
            margin-top: 20px;
        }

        p {
            color: #999999;
            text-align: center;
            margin-top: 20px;
        }

        .profile-image {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            margin: 0 auto;
            display: block;
            background-color: #cccccc;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>User Profile</h1>
        <h2>Welcome, {{ username }}!</h2>
        <p>This is the profile page for user {{ username }}.</p>
        <img class="profile-image" src="https://images.pexels.com/photos/39866/entrepreneur-startup-start-up-man-39866.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1" alt="Profile Image">
    </div>
</body>
</html>

```

Step 6: Run the Flask Server
- Add the necessary code to run the Flask server at the end of the file.
- Use the `if __name__ == '__main__':` condition to ensure that the server is only run when the script is executed directly (not imported as a module).
- Inside the condition, call the `run()` method on the Flask application (`app`).

Double-check that your work matches the solution below.
<details><summary><i>👉 Click here for the answer.</i></summary>
  
```python
if __name__ == '__main__':
    app.run()
```
</details>

Save the file.

Open a terminal or command prompt and navigate to the directory where your server.py file is located.

Run the Flask server by executing the following command:

```bash
python3 server.py
```
You should see output

::page{title="__Step 7:__ Deploy the application on Code Engine."}

1. Change to the `final_project` directory.

    ```bash
    cd /home/project/xzceb-flask_eng_fr/final_project
    ```
  > ### Note: 
  > - Make sure `Step 6: Run the servers` runs successfully.
  > - In `requirements.txt` file, mention the flask package and it should look as shown below:
 ```
 Flask

 ```
  

2. Let’s create a Dockerfile in your project directory. Dockerfile is the blueprint for building a container image for our app. 
 

3. Create `Dockerfile` and add the following lines to your file:

    ```
    FROM python:alpine3.7
    COPY . /app
    WORKDIR /app
    RUN pip install -r requirements.txt
    EXPOSE 8080
    ENTRYPOINT [ "python" ]
    CMD [ "server.py" ]
    ```

  On the first line we are importing the Docker image `python:alpine3.7` which comes with support for Python 3. This image allows us to create Flask web applications in Python that run in a single container. We are interested in the latest version of this image available, which supports Python 3.
  
   On the next 2 lines, we copy the contents of the final_project directory we just created, into an app directory in the container image. Pretty easy, right!

  Finally, we are opening port 8080 to usage in the docker container. This will allow us to access our application later once it’s deployed to the cloud.

 

4. On the menu in your lab environment, click `Skills network tools`. Click Cloud dropdown and choose `Code Engine`. The code engine set up panel comes up. Click `Create Project`.

![](/images/Image3.png)

    

5. The code engine environment takes a while to prepare. You will see the progress status being indicated in the set up panel.


6. Once the code engine set up is complete, you can see that it is active. Click on `Code Engine CLI` to begin the pre-configured CLI in the terminal below.

![](/images/Image4.png)

You will now use the CLI to deploy the application.

7. Change to the app directory where the Dockerfile was created.


    ```bash
    cd /home/project/xzceb-flask_eng_fr/final_project
    ```

8. Now run `docker build` in the app directory and tag the image. Note that in the below command we are naming the app `flask-docker-demo`.

    ```bash
    docker build . -t us.icr.io/${SN_ICR_NAMESPACE}/flask-docker-demo
    
    ```


9. Now push the image to the namespace so that you can run it.

    ```bash
    docker push us.icr.io/${SN_ICR_NAMESPACE}/flask-docker-demo:latest
    ```
   
10. Deploy the application.

    ```bash
    ibmcloud ce application create --name flask-docker-demo --image us.icr.io/${SN_ICR_NAMESPACE}/flask-docker-demo --registry-secret icr-secret

    ```

> Please note this command will run only in a Code Engine CLI. If you didn’t follow the steps 4 to 7 to start the Code Engine CLI, you may get errors.

11. Press ctrl(Windows)/cmd(Mac) and the link that is created. Alternatively copy the link and paste it in a browser page and press enter. The `flask-docker-demo` application page will be render as given below.

![](/images/Image5.png)

## Author(s)
CF

### Other Contributor(s)

## Change Log
| Date | Version | Changed by | Change Description |
|------|--------|--------|---------|
| 2023-06-15 | 0.4 | CF  | Initial Lab |# Assignment-For-Filed
