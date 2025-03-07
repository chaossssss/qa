import os
from dotenv import load_dotenv
from docx import Document
from openai import OpenAI
import json

load_dotenv()

# client = OpenAI(
#     api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
# )


class QAGenerator:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
        )

    def chunk_text(self, text, chunk_size=5000):
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    def generate_qa(self, text_chunk):
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的数据集生成器，请根据提供的文本内容生成100个高质量的问答对。问题要覆盖核心知识点，答案需准确简洁。",
                },
                {"role": "user", "content": text_chunk},
            ],
            temperature=0.3,
        )
        # print(response.choices[0].message.content)
        return response.choices[0].message.content

    def process_docx(self, file_path):
        doc = Document(file_path)
        full_text = "\n".join([para.text for para in doc.paragraphs])
        chunks = self.chunk_text(full_text)
        dataset = []
        for chunk in chunks:
            print("chunk", chunk)
            try:
                qa_content = self.generate_qa(chunk)
                print("qa_content", qa_content)
                qa_pairs = [
                    pair.split("答：") for pair in qa_content.split("\n\n") if pair
                ]
                dataset.extend(
                    [
                        {
                            "instruction": q[0].replace("问：", "").strip(),
                            "output": q[1].replace("答：", "").strip(),
                        }
                        for q in qa_pairs
                        if len(q) == 2
                    ]
                )
                print(dataset)
            except Exception as e:
                print(f"处理区块时发生错误: {str(e)}")

        with open("qa_dataset.json", "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    generator = QAGenerator()
    generator.process_docx(os.path.abspath("input.docx"))
