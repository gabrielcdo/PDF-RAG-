import os
from langchain_openai import ChatOpenAI
from utils.functions import create_retriever, create_rag_chain

# Set up your API keys
os.environ["OPENAI_API_KEY"] = ""

qa_pairs = [
    # FÁCEIS
    {
        "question": "Quais são os dois principais tipos de tarefas em que a IA foi mais utilizada, segundo o estudo?",
        "answer": "Tarefas de desenvolvimento de software e escrita."
    },
    {
        "question": "Qual percentual das ocupações mostrou uso de IA em pelo menos 25% de suas tarefas?",
        "answer": "Aproximadamente 36% das ocupações."
    },
    {
        "question": "O estudo utiliza dados de qual plataforma de IA conversacional?",
        "answer": "Claude.ai, da Anthropic."
    },
    {
        "question": "A maioria dos usos de IA identificados é de automação ou de aumento (augmentação) das capacidades humanas?",
        "answer": "Augmentação, com 57% das interações."
    },
    {
        "question": "O estudo se baseia em qual base de dados ocupacional dos EUA para mapear tarefas?",
        "answer": "A base de dados O*NET do Departamento do Trabalho dos EUA."
    },

    # MÉDIAS
    {
        "question": "Como a distribuição do uso da IA se relaciona com os salários das ocupações analisadas?",
        "answer": "O uso da IA atinge o pico no quartil superior de salários, mas é baixo nas faixas mais altas e mais baixas."
    },
    {
        "question": "Quais ocupações mostraram menor uso de IA, segundo a análise?",
        "answer": "Ocupações que envolvem manipulação física do ambiente, como trabalhadores da construção, anestesiologistas e trabalhadores agrícolas."
    },
    {
        "question": "Quais foram as três habilidades cognitivas mais frequentemente associadas ao uso da IA?",
        "answer": "Escrita, compreensão de leitura e pensamento crítico."
    },
    {
        "question": "O que representa a 'zona de trabalho' (Job Zone) usada no estudo, e qual delas apresentou maior uso da IA?",
        "answer": "Representa o nível de preparo necessário para uma ocupação. A Job Zone 4 (preparo considerável, como graduação) teve maior uso."
    },
    {
        "question": "Qual é a diferença entre os padrões de uso automativo e aumentativo no estudo?",
        "answer": "Automativo: IA executa a tarefa com pouca intervenção humana. Augmentativo: IA colabora com o usuário, refinando ou explicando a tarefa."
    },

    # DIFÍCEIS
    {
        "question": "Explique como o sistema Clio foi utilizado para classificar as tarefas com base nas conversas.",
        "answer": "Clio analisa conversas anonimizadas e as classifica com base em uma hierarquia de tarefas da O*NET, usando LLMs para associar cada conversa à tarefa mais relevante."
    },
    {
        "question": "Qual foi a abordagem hierárquica usada para mapear tarefas da O*NET às conversas?",
        "answer": "As tarefas foram organizadas em uma estrutura hierárquica de três níveis, usando embeddings, clustering e prompts para permitir classificação escalável."
    },
    {
        "question": "Por que o uso de IA foi menor em ocupações com altos salários, como médicos, segundo os autores?",
        "answer": "Devido a requisitos físicos, barreiras regulatórias e complexidade especializada, que limitam a aplicabilidade atual da IA."
    },
    {
        "question": "Como a IA foi utilizada de forma diferente entre os modelos Claude 3 Opus e Claude 3.5 Sonnet?",
        "answer": "Claude 3.5 Sonnet foi mais usado para tarefas técnicas e de codificação, enquanto Claude Opus teve maior uso em criação de conteúdo, educação e escrita criativa."
    },
    {
        "question": "Quais são as principais limitações metodológicas que o estudo reconhece ao analisar o uso real de IA no trabalho?",
        "answer": "O estudo não observa como os outputs da IA são usados, depende da base O*NET que é estática, considera apenas dados do Claude.ai e pode haver ruído na classificação feita por IA."
    }
]

def main():
    retriever = create_retriever(
        persist_directory="./indexes/chroma_db", docstore_path="./indexes/docstore"
    )
    model = ChatOpenAI(model="gpt-4o-mini")
    chain = create_rag_chain(retriever, model)

    with open("comparativo_rag_vs_gpt.txt", "w", encoding="utf-8") as f:
        for i, qa in enumerate(qa_pairs, 1):
            response = chain.invoke(qa["question"])
            f.write("_____________________________________________\n")
            f.write(f"Pergunta:\n{qa['question']}\n\n")
            f.write(f"Resposta ChatGPT:\n{qa['answer']}\n\n")
            f.write(f"Resposta RAG:\n{response}\n")
            f.write("_____________________________________________\n\n")

    print("Arquivo 'comparativo_rag_vs_gpt.txt' gerado com sucesso!")

if __name__ == "__main__":
    main()
