# running dis-nft-project:

Assumes a working Python 3 installation (with python=python3 and pip=pip3).

(1) Run the code below to install the dependencies.
>$ pip install -r requirements.txt

(2) Initialize the database, by running the SQL files (Creating the necessary tables).
    by pasting it into your Pgadmin 4 Query Tool in our database.

(3)Configure the database connection
    In the app.py file, set your own database username and password in the get_movies_by_emotion and search_movies functions:

(4) Run Web-App
>$ python app.py


----------------------------------------------------------------------------------------------

# How to use the application

(1) Frontpage
    Upon navigating to the root URL (http://localhost:5000), you'll see the frontpage where you can input your mood or search for movies.

(2) Mood Input
    Enter a statement describing your day in the provided textarea and press the "Submit" button.
    The application will analyze your statement using a Bi-directional LSTM model to predict your emotion and recommend a movie that matches your mood.
    The recommended movie details will be displayed below the form.

(3) Movie Search
    Use the search bar to search for movies in the database.
    Enter a keyword or phrase and press the "Search" button.
    The search results will be displayed on a new page, listing the movies that match your query.

(4) Movie Details
    Each recommended or searched movie will display the following details:
    Title: The title of the movie.
    Reason: Why the movie is associated with the predicted emotion.
    Description: A brief description of the movie.
    IMDB Rating: The IMDB rating of the movie.