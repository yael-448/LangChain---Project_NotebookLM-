import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# ── טוענים את המפתחות הסודיים מקובץ .env ──
load_dotenv()

# ── חלק 1: המוח (המודל) ──
model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# thinking_budget נתמך רק במודלים מסדרת 2.5 — מוסיפים אותו רק אז
model_kwargs = {"model": model_name, "temperature": 0}
if "2.5" in model_name:
    model_kwargs["thinking_budget"] = 0

model = ChatGoogleGenerativeAI(**model_kwargs)

# ── חלק 2: כלי החיפוש של Tavily ──
search_tool = TavilySearch(max_results=5)


# ── חלק 3: כלי משלנו — הגשת המקורות למשתמש לאישור ──
@tool
def present_sources(sources: list[dict]) -> str:
    """הצג למשתמש את רשימת המקורות שנאספו, לאישור או סינון.
    יש לקרוא לכלי הזה פעם אחת בלבד, אחרי שאספת מספיק מקורות מהחיפוש.

    Args:
        sources: רשימת מקורות. כל מקור הוא מילון עם המפתחות:
                 'title' (כותרת), 'url' (כתובת), 'description' (תיאור קצר).
    """
    titles = "\n".join(f"- {s.get('title')}" for s in sources)
    return (
        f"המשתמש אישר את {len(sources)} המקורות הבאים:\n{titles}\n\n"
        "כעת כתוב סיכום ראשוני בעברית המבוסס על המקורות שאושרו, "
        "כדי לתת למשתמש תמונה כללית על הנושא."
    )


# ── חלק 4: ה-System Prompt — מגדיר ל-Agent את התפקיד והתהליך ──
SYSTEM_PROMPT = """אתה עוזר מחקר שתפקידו לאסוף מקורות מידע איכותיים מהאינטרנט על נושא שהמשתמש בוחר.

תהליך העבודה שלך:
1. נתח את הנושא שהמשתמש נתן.
2. הרץ כמה שאילתות חיפוש שונות ומגוונות (לפחות 2-3) כדי לכסות את הנושא מזוויות שונות.
3. אסוף את כל המקורות הרלוונטיים שמצאת.
4. קרא פעם אחת לכלי present_sources והעבר לו את רשימת כל המקורות שאספת.
   לכל מקור ציין: title (כותרת), url (כתובת), description (תיאור קצר במשפט-שניים).
5. אחרי שהמשתמש בוחר אילו מקורות לאשר, כתוב סיכום ראשוני בעברית של התוכן
   מהמקורות שאושרו, כך שהמשתמש יקבל תמונה כללית על הנושא.

ענה תמיד בעברית, בצורה ברורה ומסודרת."""


# ── חלק 5: Checkpointer — זיכרון ל-Agent ──
checkpointer = InMemorySaver()


# ── חלק 6: בניית ה-Agent (עם HITL + זיכרון) ──
agent = create_agent(
    model=model,
    tools=[search_tool, present_sources],
    system_prompt=SYSTEM_PROMPT,
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"present_sources": True})],
    checkpointer=checkpointer,
)


# ── חלק 7: הרצה מהטרמינל עם לולאת Human-in-the-loop ──
if __name__ == "__main__":
    topic = input("📚 על איזה נושא לאסוף מקורות? ").strip()

    if not topic:
        print("⚠️ לא הוקלד נושא. נסי שוב והקלידי נושא לפני Enter.")
        raise SystemExit

    # thread_id מזהה את ה"שיחה" עבור הזיכרון (Checkpointer)
    config = {"configurable": {"thread_id": "terminal-session"}}

    print("\n🔎 ה-Agent מחפש מקורות... (רגע בבקשה)")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": topic}]}, config
    )

    # האם ה-Agent עצר וביקש אישור על המקורות?
    interrupt = result.get("__interrupt__")
    if interrupt:
        sources = interrupt[0].value["action_requests"][0]["args"]["sources"]

        print("\n" + "=" * 50)
        print(f"📋 מצאתי {len(sources)} מקורות. אילו לשמור?\n")
        for i, s in enumerate(sources, start=1):
            print(f"[{i}] {s.get('title')}")
            print(f"    {s.get('url')}")
            print(f"    {s.get('description')}\n")

        choice = input(
            "הקלידי מספרים מופרדים בפסיק לשמירה (למשל 1,3,4), "
            "'all' לשמירת הכול, או Enter לביטול: "
        ).strip().lower()

        if choice == "all":
            decision = {"type": "approve"}
        elif choice == "":
            decision = {"type": "reject", "message": "המשתמש לא בחר אף מקור."}
        else:
            keep_idx = [int(x) - 1 for x in choice.split(",") if x.strip().isdigit()]
            kept = [sources[i] for i in keep_idx if 0 <= i < len(sources)]
            decision = {
                "type": "edit",
                "edited_action": {"name": "present_sources", "args": {"sources": kept}},
            }

        print("\n📝 כותב סיכום על בסיס המקורות שבחרת...")
        result = agent.invoke(Command(resume={"decisions": [decision]}), config)

    print("\n" + "=" * 50)
    print(result["messages"][-1].text)
