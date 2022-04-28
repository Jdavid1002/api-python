from textblob import TextBlob
from googletrans import Translator
from fastapi import FastAPI
from pydantic import BaseModel
from clustering import print_clusters

class Item(BaseModel):
  text: str
  idiom: str

class TextTranslate(BaseModel):
  text: str
  idiom: str


app = FastAPI()
translator = Translator()

@app.post("/items")
async def predictSentiment(item : Item):
  translation = translator.translate(item.text, dest='en', src=item.idiom)
  result = TextBlob(translation.text).sentiment.polarity
  if result > 0:
    return {"icon": "positive"}
  else :
    return {"icon": "negative"}


@app.post("/translate")
async def predictSentiment(TextTranslate : TextTranslate):
  translationEn = translator.translate(TextTranslate.text, dest='en', src=TextTranslate.idiom)
  translationFr = translator.translate(TextTranslate.text, dest='fr', src=TextTranslate.idiom)
  translationIt = translator.translate(TextTranslate.text, dest='it', src=TextTranslate.idiom)
  translationEs = translator.translate(TextTranslate.text, dest='es', src=TextTranslate.idiom)

  return {
    "en": translationEn.text,
    "fr": translationFr.text,
    "it": translationIt.text,
    "es": translationEs.text
  }


@app.get("/clustering")
def clustering():
  return print_clusters()
