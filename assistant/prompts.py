from datetime import datetime
from langchain.prompts.prompt import PromptTemplate

# ============================================================================
# Claude basic chatbot prompt construction
# ============================================================================
#以下是Human和AI助手之間友好對話的內容。
#AI以禮貌且準確地回答問題，並在相關時提供具體細節。
#如果AI不知道某個問題的答案，它會真實地說出自己不知道。

date_today = str(datetime.today().date())

_CALUDE_PROMPT_TEMPLATE = f"""

先前已經發生的對話如下：
<conversation_history>
{{history}}
</conversation_history>

你是一個能源管理專家，你要負責制定和實施能源效率計畫，減少能源消耗，評估建築物的能源使用並提出改進方案。同時，你也是能源審計師，分析和評估能源使用情況，提出節能建議。
現在使用者會針對你負責的任務詢問你，請簡短的回答使用者。以下是使用者的問題：
<human_reply>
{{input}}
</human_reply>


如果你是AI助手你會怎麼回答：
"""

CLAUDE_PROMPT = PromptTemplate(
    input_variables=["history", "context", "input"], template=_CALUDE_PROMPT_TEMPLATE
)

## Placeholder for lab 3 - agent prompt code
## replace this placeholder with code from lab 3, step 2 as instructed.


# 以下是與使用者輸入相關的上下文：內容
# <context>
# {{context}}
# </context>

_DOC_PROMPT_TEMPLATE = f"""

從文件中查詢到的相關資料:
<context>
{{context}}
</context>

你是一個能源管理專家，你要負責制定和實施能源效率計畫，減少能源消耗，評估建築物的能源使用並提出改進方案。同時，你也是能源審計師，分析和評估能源使用情況，提出節能建議。
現在使用者會針對你負責的任務詢問你，請簡短的回答使用者。以下是使用者的問題：
<human_reply>
{{input}}
</human_reply>


如果你是AI助手你會怎麼回答：
"""
DOC_PROMPT = PromptTemplate(
    input_variables=["context", "input"], template=_DOC_PROMPT_TEMPLATE
)