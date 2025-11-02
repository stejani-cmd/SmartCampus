CREATE TABLE courses (
    id SERIAL PRIMARY KEY,
    term VARCHAR(50) NOT NULL,
    title VARCHAR(100) NOT NULL,
    details VARCHAR(255) NOT NULL,
    hours INT NOT NULL,
    crn INT NOT NULL,
    schedule_type VARCHAR(50) NOT NULL,
    grade_mode VARCHAR(50) NOT NULL,
    level VARCHAR(50) NOT NULL,
    part_of_term VARCHAR(50) NOT NULL,
);
