import numpy as np
import csv

#The following part should create mappings between country codes and country names

country_codes_file = open("additional/country_codes.csv")
countries_csv = csv.reader(country_codes_file, delimiter = ',')

country_to_code = dict()
code_to_country = dict()

line_count = 0
for row in countries_csv:
    line_count += 1
    if line_count == 1:
        continue
    country_name = row[0]
    country_code = row[1]
    country_to_code[country_name] = country_code
    code_to_country[country_code] = country_name

hashes_to_id = dict()
id_to_hashes = dict()

appearing_countries = set()
good_phones_country_count = 0
bad_phones_country_count = 0

numbers_file = open("task_and_samples/phone_numbers.csv")
traffic_file = open("task_and_samples/voice_traffic.csv")

numbers_csv = csv.DictReader(numbers_file, delimiter = ',')

line_count = 0
for row in numbers_csv:
    line_count += 1
    if line_count == 1:
        continue
    country_code = row["OPERATOR_COUNTRY_ISO2"]
    if country_code in code_to_country:
        appearing_countries.add(country_code)
        good_phones_country_count += 1
    else:
        bad_phones_country_count += 1

print("Good " + str(good_phones_country_count))
print("Bad " + str(bad_phones_country_count))
print("Different " + str(country_code.size()))

#class PhoneNumber:
#    def __init__(self, id, country, type, blacklist, a2p, )
