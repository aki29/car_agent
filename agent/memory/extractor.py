from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

extraction_prompt = PromptTemplate.from_template("""
請從以下句子中抽取任何有意義的個人資訊，並以合法 JSON 格式輸出，例如：
{{{{"name": "...", "music_preference": "...", "location": "..."}}}}
只能輸出一行合法 JSON，請勿加上多餘文字，否則會造成解析錯誤。
若找不到資訊，請回傳：{{{{}}}}

輸入句子：{text}
""")

def extract_memory_kv_chain(model):
    return extraction_prompt | model | JsonOutputParser()





