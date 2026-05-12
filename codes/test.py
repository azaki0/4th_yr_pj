import re

emergency_keywords = ["heatstroke","unconscious","not breathing","severe bleeding","chest pain","seizure","stroke"]
pattern = r"\b(" + "|".join(emergency_keywords) + r")\b"

text = "The patient is unconscious and has chest pain."
matches = re.findall(pattern, text, re.IGNORECASE)
patient_status = 'The patient status: '
for i in matches:
    patient_status =patient_status+ i + ","
print(patient_status[:len(patient_status)-1])


#SELECT file_name, page_number, content FROM medical_knowledge ORDER BY embedding <=> %s::vector LIMIT %s''', (query_embedding, n_results)