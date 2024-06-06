-- create_database.sql

-- Create the Database
CREATE DATABASE moviematchmaker;

-- Connect to the Database
\c moviematchmaker;

-- Create Tables
CREATE TABLE Emotions (
    emotion_id SERIAL PRIMARY KEY,
    emotion_name VARCHAR(50)
);

CREATE TABLE Movies (
    movie_id SERIAL PRIMARY KEY,
    title VARCHAR(255),
    reason TEXT,
    description TEXT,
    imdb_rating DECIMAL(2, 1),
    emotion_id INT,
    FOREIGN KEY (emotion_id) REFERENCES Emotions(emotion_id)
);

CREATE TABLE Users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE
);

-- Insert Data into Emotions
INSERT INTO Emotions (emotion_name) VALUES
('Sadness'),
('Joy'),
('Love'),
('Anger'),
('Fear'),
('Surprise');

-- Insert Data into Movies
INSERT INTO Movies (title, reason, description, imdb_rating, emotion_id) VALUES
('The Room', 'Watching something so notoriously bad can make you laugh and forget your own sadness.', 'A melodramatic disasterpiece about a love triangle that makes no sense.', 3.7, 1),
('Birdemic: Shock and Terror', 'The terrible special effects and wooden acting will definitely lift your spirits.', 'A horror-romance about birds attacking a small town.', 1.8, 1),
('Battlefield Earth', 'Laughing at John Travolta''s over-the-top performance is a great distraction.', 'Aliens conquer Earth, and the story follows the human rebellion.', 2.5, 1),
('Gigli', 'Often cited as one of the worst movies ever, it''s so bad it''s funny.', 'A romantic comedy with a convoluted plot and miscast actors.', 2.6, 1),
('Cats', 'The bizarre CGI and awkward musical numbers provide unintentional comedy.', 'A live-action adaptation of the famous musical that went horribly wrong.', 2.8, 1),

('Ferris Bueller''s Day Off', 'A feel-good classic about enjoying life to the fullest.', 'A high school slacker skips school for a day of fun in Chicago.', 7.8, 2),
('The Grand Budapest Hotel', 'Its quirky humor and delightful characters make it a joy to watch.', 'A comedy about a legendary concierge at a famous European hotel.', 8.1, 2),
('Superbad', 'Hilarious coming-of-age comedy that will keep you laughing throughout.', 'Two high school friends try to make the most of their last days before graduation.', 7.6, 2),
('Mamma Mia!', 'Upbeat musical numbers and a fun storyline based on ABBA''s songs.', 'A bride-to-be tries to find her real father using her mother’s old diaries.', 6.4, 2),
('Paddington 2', 'Pure-hearted and charming, it’s a delightful watch for all ages.', 'Paddington the bear ends up in prison and spreads joy and marmalade.', 7.8, 2),

('War of the Roses', 'A dark comedy about a couple’s bitter divorce.', 'A married couple tries to get divorced but ends up in a vicious battle over their home.', 6.8, 3),
('Fatal Attraction', 'A thriller that turns an affair into a nightmare, a cautionary tale about love gone wrong.', 'A man’s casual affair turns into a dangerous obsession.', 6.9, 3),
('Gone Girl', 'A thrilling take on a marriage with deep, dark secrets.', 'A man becomes the prime suspect in the disappearance of his wife.', 8.1, 3),
('500 Days of Summer', 'A realistic and often funny take on modern relationships.', 'A young man recounts his failed relationship with the girl of his dreams.', 7.7, 3),
('Blue Valentine', 'A raw and emotional look at the dissolution of a marriage.', 'The story of a couple seen through the lens of their beginning and end.', 7.4, 3),

('Happy Gilmore', 'Watching Adam Sandler’s antics on a golf course is a great way to blow off steam.', 'A rejected hockey player becomes a successful golfer to save his grandmother''s house.', 7.0, 4),
('Anger Management', 'A comedy directly about managing anger in the funniest ways possible.', 'After a misunderstanding, a mild-mannered man is sentenced to an anger management program.', 6.2, 4),
('Billy Madison', 'Adam Sandler’s goofy comedy about an adult going back to school is sure to lighten the mood.', 'A man-child goes back to grade school to prove himself worthy of his family’s fortune.', 6.4, 4),
('Grown Ups', 'A fun ensemble cast in ridiculous situations makes for a good laugh.', 'Five childhood friends reunite for a Fourth of July holiday weekend.', 6.0, 4),
('The Waterboy', 'Adam Sandler’s portrayal of a socially awkward waterboy turned football star is hilarious.', 'A waterboy for a college football team discovers he has a unique tackling ability and becomes a star.', 6.1, 4),

('Scary Movie', 'A parody of horror films that turns fear into laughs.', 'A group of teenagers find themselves being stalked by a dim-witted killer.', 6.2, 5),
('Shaun of the Dead', 'A comedic take on a zombie apocalypse, blending fear with humor.', 'A man decides to turn his life around by winning back his ex and fighting zombies.', 7.9, 5),
('Tucker and Dale vs. Evil', 'Misunderstandings and slapstick comedy in a horror setting make for a funny movie.', 'Two hillbillies are mistaken for killers by a group of clueless college students.', 7.5, 5),
('The Cabin in the Woods', 'A self-aware horror-comedy that deconstructs the genre in a humorous way.', 'Five friends go to a remote cabin, where they get more than they bargained for.', 7.0, 5),
('Zombieland', 'A comedic and action-packed take on surviving the zombie apocalypse.', 'A shy student trying to reach his family in Ohio teams up with three strangers to survive.', 7.6, 5),

('Kiss Kiss Bang Bang', 'A twisty, humorous neo-noir with plenty of surprises.', 'A murder mystery brings together a private eye, a struggling actress, and a thief pretending to be an actor.', 7.5, 6),
('The Nice Guys', 'Unexpected plot twists and witty banter make for a surprising comedy.', 'A private eye and a tough enforcer team up to investigate the disappearance of a young woman.', 7.4, 6),
('Hot Fuzz', 'A police comedy with surprising turns and plenty of laughs.', 'A top London cop is transferred to a small town with a dark secret.', 7.8, 6),
('Game Night', 'A comedy with unexpected plot twists that keeps you guessing.', 'A group’s game night turns into a real-life mystery.', 6.9, 6),
('Burn After Reading', 'A dark comedy with a series of unexpected events.', 'A disk containing the memoirs of a CIA agent ends up in the hands of two gym employees.', 7.0, 6);
