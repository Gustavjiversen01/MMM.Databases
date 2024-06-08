# running dis-MMM-project:

Assumes a working Python 3 installation (with python=python3 and pip=pip3).

(1) Navigate to your MMM.databases folder and open it in visual studio code. 

(2) Navigate to the MMM.databases folder in the terminal and run the code below to install the dependencies.
>$ pip install -r requirements.txt

(3) Initialize the database, by running the SQL files (Creating the necessary tables).
    by pasting it into your Pgadmin 4 Query Tool in a new database.
    (If you have problems with premission, try this command: GRANT ALL PRIVILEGES ON DATABASE your_database TO your_username;)
    (Replace your_database with your database and your_username with your username)

(4)Configure the database connection
    In the app.py file, set your own database username and password in the get_movies_by_emotion and search_movies functions:

(5) Run Web-App
>$ python app.py


----------------------------------------------------------------------------------------------

# How to use the application

(1) Frontpage\\
    Upon navigating to the root URL (http://localhost:5000), you'll see the frontpage where you can input your mood or search for movies.

(2) Mood Input\\
    Enter a statement describing your day in the provided textarea and press the "Submit" button.
    The application will analyze your statement using a Bi-directional LSTM model to predict your emotion and recommend a movie that matches your mood.
    The recommended movie details will be displayed below the form.

(3) Movie Search\\
    Use the search bar to search for movies in the database.
    Enter a keyword or phrase and press the "Search" button.
    The search results will be displayed on a new page, listing the movies that match your query.

(4) Movie Details\\
    Each recommended or searched movie will display the following details:
    Title: The title of the movie.
    Reason: Why the movie is associated with the predicted emotion.
    Description: A brief description of the movie.
    IMDB Rating: The IMDB rating of the movie.