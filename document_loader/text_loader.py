from langchain_community.document_loaders import TextLoader

loader = TextLoader('files/myinfo.txt', encoding='utf8')

documents = loader.load()

# print(documents)

# print(documents[0].page_content)

print(documents[0].metadata)
