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
('Troll 2', 'Infamously bad acting and plot make it hilariously entertaining.', 'A family vacation turns into a nightmare when they discover their destination is overrun by goblins.', 2.9, 1),
('Plan 9 from Outer Space', 'Often cited as one of the worst movies ever made, it’s so bad it’s funny.', 'Aliens resurrect dead humans as zombies and vampires to stop humanity from creating a doomsday weapon.', 4.0, 1),
('Manos: The Hands of Fate', 'A cult classic for its utter incompetence in filmmaking.', 'A family is trapped in a lodge by a polygamous cult leader.', 1.9, 1),
('Howard the Duck', 'A bizarre concept and poor execution make it a funny watch.', 'A humanoid duck is pulled from his homeworld to Earth, where he must stop an alien invader.', 4.7, 1),
('Mac and Me', 'Notorious for being a blatant E.T. rip-off with unintentional humor.', 'An alien tries to evade capture while befriending a young boy in a wheelchair.', 3.4, 1),

('Ferris Bueller''s Day Off', 'A feel-good classic about enjoying life to the fullest.', 'A high school slacker skips school for a day of fun in Chicago.', 7.8, 2),
('The Grand Budapest Hotel', 'Its quirky humor and delightful characters make it a joy to watch.', 'A comedy about a legendary concierge at a famous European hotel.', 8.1, 2),
('Superbad', 'Hilarious coming-of-age comedy that will keep you laughing throughout.', 'Two high school friends try to make the most of their last days before graduation.', 7.6, 2),
('Mamma Mia!', 'Upbeat musical numbers and a fun storyline based on ABBA''s songs.', 'A bride-to-be tries to find her real father using her mother’s old diaries.', 6.4, 2),
('Paddington 2', 'Pure-hearted and charming, it’s a delightful watch for all ages.', 'Paddington the bear ends up in prison and spreads joy and marmalade.', 7.8, 2),
('Anchorman: The Legend of Ron Burgundy', 'Absurd humor and quotable lines from Will Ferrell’s iconic performance.', 'The adventures of a clueless but charming news anchor in the 1970s.', 7.2, 2),
('Elf', 'A heartwarming and funny holiday movie starring Will Ferrell.', 'A human raised by elves at the North Pole travels to New York to find his real father.', 7.0, 2),
('The Princess Bride', 'A whimsical and witty fairy tale that delights all ages.', 'A young woman and her true love must overcome various obstacles to be together.', 8.1, 2),
('Monty Python and the Holy Grail', 'Classic British humor with absurd and hilarious scenes.', 'King Arthur and his knights embark on a surreal quest for the Holy Grail.', 8.2, 2),
('School of Rock', 'Jack Black’s energetic performance and the feel-good storyline.', 'A failed musician poses as a substitute teacher and turns his class into a rock band.', 7.2, 2),

('War of the Roses', 'A dark comedy about a couple’s bitter divorce.', 'A married couple tries to get divorced but ends up in a vicious battle over their home.', 6.8, 3),
('Fatal Attraction', 'A thriller that turns an affair into a nightmare, a cautionary tale about love gone wrong.', 'A man’s casual affair turns into a dangerous obsession.', 6.9, 3),
('Gone Girl', 'A thrilling take on a marriage with deep, dark secrets.', 'A man becomes the prime suspect in the disappearance of his wife.', 8.1, 3),
('500 Days of Summer', 'A realistic and often funny take on modern relationships.', 'A young man recounts his failed relationship with the girl of his dreams.', 7.7, 3),
('Blue Valentine', 'A raw and emotional look at the dissolution of a marriage.', 'The story of a couple seen through the lens of their beginning and end.', 7.4, 3),
('Revolutionary Road', 'A raw and honest depiction of a troubled marriage.', 'A couple in the 1950s struggles with their personal ambitions and societal expectations.', 7.3, 3),
('Marriage Story', 'An emotional yet humorous take on the complexity of divorce.', 'A stage director and his actor wife struggle through a grueling divorce that pushes them to their limits.', 7.9, 3),
('Closer', 'A realistic and sometimes brutal look at modern relationships.', 'The tangled love lives of four people in London.', 7.2, 3),
('The Break-Up', 'A comedy that humorously explores the end of a relationship.', 'A couple’s breakup becomes complicated when neither wants to move out of their shared apartment.', 5.8, 3),
('Unfaithful', 'A suspenseful drama about infidelity and its consequences.', 'A woman’s affair leads to unexpected and tragic outcomes.', 6.7, 3),

('Happy Gilmore', 'Watching Adam Sandler’s antics on a golf course is a great way to blow off steam.', 'A rejected hockey player becomes a successful golfer to save his grandmother''s house.', 7.0, 4),
('Anger Management', 'A comedy directly about managing anger in the funniest ways possible.', 'After a misunderstanding, a mild-mannered man is sentenced to an anger management program.', 6.2, 4),
('Billy Madison', 'Adam Sandler’s goofy comedy about an adult going back to school is sure to lighten the mood.', 'A man-child goes back to grade school to prove himself worthy of his family’s fortune.', 6.4, 4),
('Grown Ups', 'A fun ensemble cast in ridiculous situations makes for a good laugh.', 'Five childhood friends reunite for a Fourth of July holiday weekend.', 6.0, 4),
('The Waterboy', 'Adam Sandler’s portrayal of a socially awkward waterboy turned football star is hilarious.', 'A waterboy for a college football team discovers he has a unique tackling ability and becomes a star.', 6.1, 4),
('Big Daddy', 'Adam Sandler’s character’s struggle to grow up while taking care of a child is both touching and funny.', 'A lazy law school grad adopts a kid to impress his girlfriend, but everything doesn’t go as planned.', 6.4, 4),
('Mr. Deeds', 'Sandler’s portrayal of a kind-hearted but naive small-town man is endearing and amusing.', 'A small-town guy inherits a fortune from his uncle and moves to the big city.', 5.8, 4),
('Just Go with It', 'A light-hearted romantic comedy with Sandler’s typical humor.', 'A plastic surgeon convinces his assistant to pose as his soon-to-be-divorced wife to cover up a lie.', 6.4, 4),
('You Don’t Mess with the Zohan', 'A ridiculous and over-the-top comedy about an Israeli commando turned hairdresser.', 'A former counter-terrorist becomes a hairstylist in New York.', 5.5, 4),
('Click', 'A funny yet thought-provoking film about appreciating life’s moments.', 'A workaholic architect finds a universal remote that allows him to fast-forward and rewind his life.', 6.4, 4),

('Scary Movie', 'A parody of horror films that turns fear into laughs.', 'A group of teenagers find themselves being stalked by a dim-witted killer.', 6.2, 5),
('Shaun of the Dead', 'A comedic take on a zombie apocalypse, blending fear with humor.', 'A man decides to turn his life around by winning back his ex and fighting zombies.', 7.9, 5),
('Tucker and Dale vs. Evil', 'Misunderstandings and slapstick comedy in a horror setting make for a funny movie.', 'Two hillbillies are mistaken for killers by a group of clueless college students.', 7.5, 5),
('The Cabin in the Woods', 'A self-aware horror-comedy that deconstructs the genre in a humorous way.', 'Five friends go to a remote cabin, where they get more than they bargained for.', 7.0, 5),
('Zombieland', 'A comedic and action-packed take on surviving the zombie apocalypse.', 'A shy student trying to reach his family in Ohio teams up with three strangers to survive.', 7.6, 5),
('Young Frankenstein', 'Mel Brooks’ hilarious take on the Frankenstein legend.', 'A young neurosurgeon inherits the castle of his grandfather, the famous Dr. Frankenstein.', 8.0, 5),
('Beetlejuice', 'A quirky and dark comedy with Tim Burton’s unique style.', 'A recently deceased couple hires a bio-exorcist to remove the new owners of their house.', 7.5, 5),
('Ghostbusters', 'A classic comedy about paranormal investigators.', 'Three parapsychologists start a ghost-catching business in New York City.', 7.8, 5),
('The Addams Family', 'A spooky yet funny film about a bizarre and endearing family.', 'The eccentric Addams family faces off against a greedy con artist.', 6.9, 5),
('Gremlins', 'A horror-comedy with mischievous creatures causing chaos.', 'A young man inadvertently breaks three important rules concerning his new pet and unleashes a horde of malevolent little monsters.', 7.3, 5),

('Kiss Kiss Bang Bang', 'A twisty, humorous neo-noir with plenty of surprises.', 'A murder mystery brings together a private eye, a struggling actress, and a thief pretending to be an actor.', 7.5, 6),
('The Nice Guys', 'Unexpected plot twists and witty banter make for a surprising comedy.', 'A private eye and a tough enforcer team up to investigate the disappearance of a young woman.', 7.4, 6),
('Hot Fuzz', 'A police comedy with surprising turns and plenty of laughs.', 'A top London cop is transferred to a small town with a dark secret.', 7.8, 6),
('Game Night', 'A comedy with unexpected plot twists that keeps you guessing.', 'A group’s game night turns into a real-life mystery.', 6.9, 6),
('Burn After Reading', 'A dark comedy with a series of unexpected events.', 'A disk containing the memoirs of a CIA agent ends up in the hands of two gym employees.', 7.0, 6),
('The Truman Show', 'A unique concept with surprising twists that keeps viewers engaged.', 'An ordinary man discovers his life is a TV show.', 8.2, 6),
('Fight Club', 'A film with a mind-bending twist and dark humor.', 'An office worker and a soap salesman form an underground fight club.', 8.8, 6),
('Ocean’s Eleven', 'A stylish and clever heist film full of surprises.', 'Danny Ocean and his eleven accomplices plan to rob three Las Vegas casinos simultaneously.', 7.7, 6),
('The Prestige', 'A film about magicians with a twisty and surprising plot.', 'Rival magicians compete to create the ultimate stage illusion.', 8.5, 6),
('Get Out', 'A horror-thriller with surprising twists and social commentary.', 'A young African-American visits his white girlfriend’s parents for the weekend, where his simmering uneasiness about their reception of him eventually reaches a boiling point.', 7.7, 6);
