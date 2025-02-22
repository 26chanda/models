import pandas as pd
import numpy as np
import random
data = []

# Define the possible values for each column
names = ["John", "Jane", "Bob", "Alice", "Mike", "Sarah", "Emily", "David", "Lily", "Oliver"]
languages = ["English", "Spanish", "French", "Mandarin", "Arabic", "Portuguese", "Russian", "Japanese", "German", "Italian"]
companies = ["Google", "Amazon", "Microsoft", "Facebook", "Apple", "IBM", "Oracle", "Cisco", "Intel", "Dell"]
job_roles = ["Machine Learning Scientist", "Data Scientist", "Data Analyst", "Business Analyst", "Business Intelligence", "Data Science"]
departments = ["Engineering", "Research", "Marketing", "Sales", "HR", "Finance", "Operations", "IT", "Executive"]
locations = ["New York", "San Francisco", "London", "Paris", "Tokyo", "Beijing", "Mumbai", "Bangalore", "Sydney", "Melbourne"]
years_of_experience = ['1', '2','3', '4', '5', '6', '7', '8', '9', '10']
modes_of_working = ["Full-time", "Part-time", "Remote", "Freelance", "Contractor"]
countries = ["USA", "UK", "Canada", "Australia", "India", "China", "Japan", "Germany", "France", "Italy"]

# Generate 600,000 rows of data
for i in range(600000):
    name = random.choice(names)
    language = random.choice(languages)
    company = random.choice(companies)
    job_role = random.choice(job_roles)
    department = random.choice(departments)
    location = random.choice(locations)
    years_of_experience = random.choice(years_of_experience)

# Define job role descriptions
job_role_descriptions = {
    "Machine Learning Scientist": {
        "Developing and deploying machine learning model", 
        "Collaborating with cross-functional teams",
        " Analyzing and improving model performance",
        "Mentoring junior team members",
        "Staying up-to-date with industry trends and technologies",
        " Designing and implementing machine learning pipelines",
        "Conducting research and development of new models",
        "Collaborating with stakeholders to drive business growth"
    },
    "Data Scientist": {
        "Developing and maintaining data pipelines",
        "Analyzing and interpreting complex data sets",
        " Collaborating with cross-functional teams",
        " Mentoring junior team members",
        " Staying up-to-date with industry trends and technologies",
        "Designing and implementing data visualization tools",
        " Conducting research and development of new data products"
        " Collaborating with stakeholders to drive business growth"
    },
    "Data Analyst": {
        " Analyzing and interpreting data sets",
        "Developing and maintaining reports and dashboards",
        " Collaborating with cross-functional teams",
        " Mentoring junior team members",
        " Staying up-to-date with industry trends and technologies",
        " Designing and implementing data visualization tools",
        " Conducting research and development of new data products",
        " Collaborating with stakeholders to drive business growth",
        "Mentoring junior team members"
    },
    "Business Analyst": {
        "Analyzing and interpreting business data",
        " Developing and maintaining business cases",
        " Collaborating with cross-functional teams",
        " Mentoring junior team members", 
        " Staying up-to-date with industry trends and technologies",
        " Designing and implementing business process improvements",
        "Conducting research and development of new business strategies",
        " Collaborating with stakeholders to drive business growth"
    },
    "Business Intelligence": {
        " Developing and maintaining business intelligence tools",
        "Analyzing and interpreting complex data sets",
        " Collaborating with cross-functional teams",
        " Mentoring junior team members",
        "Staying up-to-date with industry trends and technologies",
        " Designing and implementing data visualization tools",
        "Conducting research and development of new business intelligence products",
        " Collaborating with stakeholders to drive business growth"
    }
}
mode_of_working = random.choice(modes_of_working)
country = random.choice(countries)
    
data.append({
    "Name": name,
    "Language": language,
    "Company": company,
    "Job Role": job_role,
    "Department": department,
    "Location": location,
    "Years of Experience": years_of_experience,
    "Description ": job_role_descriptions,
    "Mode of Working": mode_of_working,
    "Country": country
})

# Convert the list to a Pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("dataset.csv", index=False)