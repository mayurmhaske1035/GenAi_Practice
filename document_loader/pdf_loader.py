from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('files/D_Pharmacy_syllabus.pdf')

docs = loader.load()

# print(docs)

# print(docs[0].page_content)

# print(docs[0].metadata)

print(docs[14])