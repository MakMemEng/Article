---
title: "生成系AI関連でよくみる’RAG’って何？"
emoji: "💭"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Azure", "OpenAI", "ChatGPT", "LangChain", "RAG"]
published: true
published_at: 2024-05-31 21:00
---
# RAGとは？

gpt-4やgpt-3.5-turboは2023年4月までのPublicな情報が学習されている。(2024年2月現在) しかし、もっと新しい情報、プライベートな情報を使わせたいことは多い。

そこで、Promptに文脈(context)を入れる方法を用いる。 文書をベクトル化してVector Storeに保存しておき、質問に関係しそうな文書を検索してPromptに含め、その内容を踏まえてLLMに回答させる。 この手法は検索拡張生成(Retrieval Augmented Generation、通称RAG)と呼ばれている。

![](/images/b87f8b1cd1edef/image.png)

## 使用例

![](/images/b87f8b1cd1edef/image(1).png)

1. `query`: 「このShellScriptを実行したときの一連の流れを説明してください。」
2. Embedding: `query`を数値に変換(Vector化)
3. Retriever: 2の結果と近しい情報を検索
4. Prompt: 「あなたはつよつよエンジニアです。初学者にも分かりやすいように回答してください。」
5. 4の指示に従い、3の内容をもとに回答生成

## 準備

```toml
python = ">=3.9.6, <3.12"
python-dotenv = "^1.0.0"
faiss-cpu = "^1.7.4"
tiktoken = "^0.5.1"
langchain = "^0.1.4"
openai = "^1.10.0"
langchain-openai = "^0.0.5"
unstructured = "^0.13.2"
```

```python
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# 環境変数の読み込み
load_dotenv()

# API設定の変数
openai_api_key = os.getenv("AZURE_OPENAI_KEY")
azure_endpoint = os.getenv("AZURE_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_VERSION")
azure_deployment = os.getenv("AZURE_OPENAI_ENGINE")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_ENGINE")

# テキストembeddingモデル
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=azure_endpoint,
    deployment=embedding_deployment,
    api_key=openai_api_key,
    api_version=api_version,
    chunk_size=1,
)

# AzureChatOpenAI
llm = AzureChatOpenAI(
    api_version=api_version,
    azure_endpoint=azure_endpoint,
    openai_api_key=openai_api_key,
    azure_deployment=azure_deployment,
    temperature=0,
)

# テキスト分割ツール
text_splitter = CharacterTextSplitter(
    separator="\r\n",
    chunk_size=512,
    chunk_overlap=128,
    length_function=len,
)

```

## 文書をVector変換してVectorStoreを作成

```python
from langchain_community.document_loaders import DirectoryLoader

# 特定のディレクトリにある特定のファイルを文書として使用
loader_sh = DirectoryLoader("./.../bash", glob="**/*.sh")

# 一度に大量の文書をLLMに渡すとToken制限を超えるため文書を複数のchunkに分割
docs_sh = loader_sh.load_and_split()

# VectorStore作成
index = FAISS.from_documents(docs_sh, embeddings)
```

```python
# 作成したVectorStoreをローカルに保存することで、既に変換済みのVectorStoreを再利用できる
index.save_local("./sample_index")

# ローカルに保存したVectorStoreを読み込む
index = FAISS.load_local("./sample_index/", embeddings)
```

## Promptを作成して、LLMに質問を投げる

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

template = """Answer the question based on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
retriever = index.as_retriever()
output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
chain = setup_and_retrieval | prompt | llm | output_parser
```

```python
query = "○○.shの構成と実行の一連の流れを説明してください。また、各要素の参照コードも教えてください。"
result = chain.invoke(query)
print(result)
```

### 参考

<https://python.langchain.com/docs/expression_language/get_started>
