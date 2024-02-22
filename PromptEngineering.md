## How to write effecient and powerful prompts

#### Data Visualization NER keyword identification use-case 


This is an on-going and initial phase of some of the work I am currently doing. The documentation might not really have a fixed order or might be messy - I apologize for that!

From knowing nothing about prompt engineering - this is my learning process over the last 6 months.

# Phase 1

#### Problem definition 

Identify keywords related to data visualization in journal articles. 

#### OpenAI - API setup

To access and use GPT 3.5 API - get your own unique and secret key by signing up and setting up a key for your project here - https://platform.openai.com/api-keys

Make sure to include this at the start of every file or project


```python
import os
os.environ['OPENAI_API_KEY'] = "ABCD" # replace with your own API Key
```

#### GPT-3.5-turbo

- For our initial POC I utilized gpt-3.5-turbo, which is a pretty powerful LLM. 
- It has sufficient amounts of token-limits per hour and is friendly for performing testing. 
- You can send about 4098 tokens to gpt-3.5-turbo in each iteration.

#### Prompt

Prompt is the message or instruction that you provide to a LLM. The more defined, clear and consice it is - the better the results are. Our initial prompt looked like :


```python
prompt_template = """Act like a Natural Language Processing model for relevant keyword extraction. 
                    I want you to identify engineering/AI/tech keywords in the transcript and 
                    return them seperated by comma.
                    Question: {question}
                   """
```

#### Results

The keywords generated were large in number, and very generic. A more robust and fine-tuned approach had to be sough after.

#### Next Steps to consider

- Have a database of keywords that are technology related - 1000 words, 2000 words, X words. Have a LLM model reference this dataset - Retrieval Augmentation.
- TF-IDF is not ideal since the transcipts have many unique words, will be very sparse. Can try topic modelling, but we are assuming that there are only technology/humanities related keywords.

# Phase 2

#### Main Goal

A dataset consisting of 100 words related to data visualization was sourced. This dataset would now be a source of context or additional information that we would provide to our LLM.

#### Retrieval Augmentation

The whole idea behind connecting a LLM to an additional data source or a "vector-store" is called Retrieval Augmentation. In our case we connect our LLM to a vector database(pinecone database) which consisted of vector embeddings of the data visualization dataset that we sourced.

The LLM now will make its decisions based on the following:     
**prompt + the data it is trianed on + the vector store it is referencing(context)**


```python
prompt_template = """Identify data visualization related keywords in the input. Return them seperated by comma. 
                     Vector store consists of keywords related to data visualization. 
                     Use this as context to identify data visualization related keywords.

                    {context}
                    Question: {question}
                  """
```

#### Results

The keywords were subjectively much better compared to our previous prompt template. Our initial RA approach was a success, it was now about how to limit the keywords that do not make a lot of sense.

#### Next Steps to consider

- Reduce noisy keywords that are generated
- Assign a similarity score with the input data-viz dataset with the keywords generated, and enforce a minimum threshold

# Phase 3

#### Few-shot learning

Adding a few labeled examples to the LLM - to give it some instruction, context and direction.    
This is what a typical few-shot learning prompt would look like:

1. Task you want the LLM to perform
2. A brief description of what you want the ML to do
3. Few-shot examples of what the task looks like
4. Your input


```python
technology_template = """Task: Technology related keyword identification. 

Please return keywords that are related to technology in the transcript. 
Output should only contain keywords seperated by comma as shown in the examples below. 
If there are no keywords return an empty string.


EXAMPLES
___________________

Transcript 1: The idea that digital art history projects can reach wider audiences and disrupt the field of colonial 
Latin American art history is one that warrants greater discussion. Here, I would like to offer a few reflections 
and discussion points about how incorporating digital art history practices into undergraduate classes about colonial 
Latin American visual culture has encouraged better digital critical thinking by encouraging students to think about 
art history’s ontological issues and nomenclature, descriptive metadata, and issues of colonialism (or neocolonialism), 
as well as the ways in which digital images have the potential to reframe how we understand the visual culture of this 
region and time period.[13]

Output 1: digital art,  metadata, digital images

Transcript 2: How can we situate OCR against or within this history? A full narrative of transcription practices that extends 
from the sixteenth to the twenty-first century is beyond the scope of this article, though we might observe certain 
continuities or breaks with the past. Already by the nineteenth century, the hand transcription of culturally valuable 
documents, for example, had shifted away from the church and towards academic institutions and libraries. Today, this 
work is carried out by faculty members, students, community volunteers, and occasionally the workers on Amazon’s so-called 
“Mechanical Turk”.[10] Most transcriptions of printed documents, however, are produced through computer-aided processing. 
This might suggest that the labor of transcription has become, at least in part, computer labor, and that the artificially 
intelligent computer may be in some ways analogous to the Franciscan friar or his indigenous students.[11]

Output 2: OCR, transcription, printed, computer-aided, artificially intelligent

__________________

What will be the output for the following transcript based on above examples?

Transcript: {question}

Output:
"""
```

#### Results

- Prompt needs more re-work and iteration. Few-shot examples can be improved or made more specific.
- Similarity with Data Viz keywords - it was not super effective. Since there was not much correlation with technology and data viz keywords. If we are checking similarity we do not need a LLM, we can just check the similarity directly with every word and filter it.

#### Next Steps to consider


- Add more conversational buffer - Add memory to the LLM, or make it prompt better results based on initial results.
- Chat GPT 4 - Needs some activation, or funds need to be added to account (0.5$) before usage.

# Phase 4

### Version 1

- A variety of different prompts were tried and tested.
- Limiting the result to 10 keywords per execution - helped in limiting the scope and imagination of the LLM.
- Limiting the result was not always consistent. Even if we mention the LLM that we want <10 keywords, at times it generated more than 10 keywords.
- We are also passing the vectorstore along with the few-shot examples for aiding the LLM.


```python
prompt_template = """Your task is to analyze a transcript and identify the top 10 keywords 
    related to data visualization that are present in the user input. If there are fewer than 10 
    keywords, return only the ones that are present. Make sure to convert all the keywords to 
    lowercase and remove any duplicates.


    EXAMPLES
    ___________________

    Transcript 1: Visualization tools allow humanists make sense of large sets of data in the form of graphs, 
    charts,   infographics, information dashboards, and more. By using quantitative data taken from artifacts 
    such as texts and maps or demographic data such as surveys and census results, humanists can support more 
    traditional types of qualitative research by embedding information visualizations into their writing 
    and presentations. Visualization tools can aid in the discovery of larger patterns related to artifacts 
    that they would not be able to see simply by looking at the data. However, the visualizations can also obscure 
    information or reinforce biases and silences in the data. Therefore, an important component of the 
    digital humanities scholarship is using a critical lens to “close read” information visualizations and 
    the datasets they depict. 

    Output 1: visualization,  graphs, charts, dashboards, maps, surveys, census

    Transcript 2: Such visual displays, including graphs and charts, may present themselves as objective 
    even unmediated views of reality, rather than a rhetorical constructs. 
    The expression of quantifiable or quantitative information in graphic form, such as bar and pie chart.
    Visualizations of data derived from large-scale data sets such as social networks, digitized corpora, 
    and demographic data.As with so many aspects of digital work, these technologies are intertwined with 
    traditional methods. Knowing what and how to read the visualized forms is at the basis of digital 
    literacy and the assessment of meaning in there new formats.

    Output 2: graphs, visual, display, charts, bar chart, pie chart, social networks, digitized corpora

    __________________
    {context}
    
    Based on the examples and the vectorstore, idetnify the top 10 keywords related to data 
    visualization in the user input. 

    Transcript: {question}
"""
```

### Version 2


#### Persona Approach 

Adding a persona to the LLM. Giving the LLM an identity, limiting the scope or enhancing the importance of given role.

##### How it works

You are X, do Y

1. You are Data Viz expert
2. You are Data viz professor
3. You write journal articles about Data Viz



```python
prompt_template = """ You are an author of a dictionary containing Data visualization keywords. 
_______________________________
Your task is to identify the top 10 data visualization related keywords present in the input transcript. 
________________________________
Return them as comma seperated values.

Transcript: {question}
"""

```

#### Results


- Prompt can keep using more and more fine-tuning. I have added multiple checks/edge case handling - if GPT gives wrong output, I check whether or not the word is actually present in the transcript or not.

- Keywords are generic, most articles do not have distinct/out-standing data viz keywords.

- Similarity analysis not efficient/useful because of size of matrix and infinite corpus of words.

#### Next Steps to Consider

- Currently using only GPT 3.5 which is powerful enough but we might get better results with GPT4.

- Enhance persona approach prompting.



# Phase 5

Adding a persona worked out to be very powerful.


### Chain-of-thought prompting

Based on the output the LLM gives us, you add the entire conversation including the output and continue the conversation

- Ask the LLM how it arrived to the answer
- Give the LLM its responses to the questions and ask it if it performed well
- Ask LLM to evalues its performance
- Ask LLM to correct or reiterate the task at hand

### Version 1

Vanilla Chain of thought prompting    
**Conversation_1**

1. You assign a role to the system or the assistant
2. You provide the assistant a sample document


```python
conversation_1 = [
    {"role": "system", "content": '''You are an author of a dictionary containing Data visualization keywords. 
    Your task is to identify Top 10 data visualization related keywords present in the input transcript. 
    - Return them as comma seperated values.'''},
    
    {"role": "user", 
    "content": ""}]
```


```python
# Sample assistant response
output_1 = [""]
```

The assistant gives you an ouput(#1)

**Conversation_2**

1. Append the conversation history soo far i.e Conversation 1
2. Append the results of the assistant i.e output(#1)
3. Ask GPT to return only the keywords that are semantically related to Data Viz



```python
conversation_2 = [
    conversation_1,
    
    {"role": "assistant", 
    "content": "output_1"},
    
    {"role": "user", 
    "content": '''Return only the keywords that are semantically related to data visualisation in Output 1
          - Return them as comma seperated values.
          - If none exist, return nothing'''}
    
]
```

### Version 2

Chain of thought with few-shot learning    
**Conversation_1**

1. You assign a role to the system or the assistant
2. You provide the assistant with a few examples about how to do the task - fewshot learning


```python
conversation_1 = [
    {"role": "system", 
     "content": '''You are an author of a dictionary containing Data visualization keywords. 
    _______________________________
    Your task is to identify Top 10 data visualization related keywords present in the input transcript. 
    ________________________________
    Return them as comma seperated values.'''},
    
    {"role": "user", 
    "content": '''Input 1 : Visualization tools allow humanists make sense of large sets of data in the form of graphs, 
    charts,   infographics, information dashboards, and more. By using quantitative data taken from artifacts 
    such as texts and maps or demographic data such as surveys and census results, humanists can support more 
    traditional types of qualitative research by embedding information visualizations into their writing 
    and presentations. Visualization tools can aid in the discovery of larger patterns related to artifacts 
    that they would not be able to see simply by looking at the data. However, the visualizations can also obscure 
    information or reinforce biases and silences in the data. Therefore, an important component of the 
    digital humanities scholarship is using a critical lens to “close read” information visualizations and 
    the datasets they depict. '''}, 
    
    {"role": "assistant",
    "content": '''Output 1 : visualization, inforgraphics,  graphs, charts, information dashboards, quantitative data, 
    qualitative research, maps, data mapping'''},
    
    {"role": "user", 
     "content": '''Input 2 : Such visual displays, including graphs and charts, may present themselves as objective 
    even unmediated views of reality, rather than a rhetorical constructs. 
    The expression of quantifiable or quantitative information in graphic form, such as bar and pie chart.
    Visualizations of data derived from large-scale data sets such as social networks, digitized corpora, 
    and demographic data.As with so many aspects of digital work, these technologies are intertwined with 
    traditional methods. Knowing what and how to read the visualized forms is at the basis of digital 
    literacy and the assessment of meaning in there new formats.'''},
    
     {"role": "assistant",
    "content": '''Output 2 : graphs, visual, display, charts, bar chart, pie chart, social networks, digitized corpora'''},
    
    {"role": "user", 
    "content": "Input 3 : keeping tradition buddhist philosophy meditation alive long died india century recent efforts digitize materials textual tradition offer opportunities broaden circulation rare materials exiled tibetan scholarly community also suggest conceptual challenges arising complexity texts inherently multimodal character paper describes scholarly meditative traditions texts come discusses possible approaches digitization describes scholarly meditative traditions buddhist sutras india come discusses possible approaches digitization thousand years tibet preserved translated ancient buddhist sutras india keeping tradition buddhist philosophy meditation alive long died india century accuracy tibetan translations sanskrit buddhist texts meant modern scholars could establish oldest versions buddhist theories reliability century scholar max müller noted tibetan version asvaghosha appears much closer original sanskrit chinese fact verbal accuracy often reproduce exact words original since certain sanskrit words always represented tibetan equivalents instance prepositions prefixed verbal roots known care translation also understanding buddhism tibet monastic scholars added commentaries original indian root texts established hundreds libraries universities texts could studied ages tibet ancient texts received careful handling storage sacred texts printed hand wood blocks wrapped silk covers stored shelves behind main altars shrine rooms monasteries temples representing one three jewels buddhism dharma texts instruct buddhists achieve enlightenment canonical texts represent one piece always multimedia enactment key meanings vajrayana buddhism particular form buddhism tibet inherited india vajrayana buddhism also known tantra emphasizes intensive meditation meditation methods incorporate complex visual imagery bodily movements sounds drums horns well chanting texts within overall container quiet meditation practice instructions visualizations bodily gestures chanting practices passed oral lineages teacher student codified sacred rituals monastics joined together learning standardizing practices today tibetan textual tradition faces extinction unless serious efforts made preserve vajrayana practices supports years following chinese invasion tibet tibetan scholars struggled preserve circulate classical texts ancient buddhist heritage many books lost chinese launched cultural revolution tibetan soil destroying libraries temples monasteries universities tibet kept books safe important texts elucidating tibet distinctive vajrayana meditation methods liturgies philosophical investigations may never found although search tibetans still continues today famous commentaries instruction manuals tibetan scholars fled india first attempted preserve texts carried printing texts without delay time paper available india poor quality paper rice paper print bled sides making texts illegible many classical tibetan texts never reprinted left original woodblock form called pecha texts see figure lack facilities poor economic conditions tibetan refugees scattered abroad introduced added risks storage paper texts simply transcribing rescued texts protecting archives stopgap measures tibetan refugees living india europe north america south america parts asia impossible establish centrally located archives tibetans could easily study texts learn highly detailed vajrayana meditation methods described texts western scholars meditators became interested learning tibetan buddhism also found difficult locate understand old tibetan texts mentioned recent commentaries indigenous tibetan scholars unable verify ancient texts actually preserved texts stored iconography complex symbolism associated vajrayana texts explained texts tibetan pecha texts image three tibetan texts late tibetan scholars recognized possibilities digital formats preserving texts circulating texts amongst members tibetan community digitized texts could made accessible tibetan western scholars around world internet digital formats also made cataloguing texts searching specific titles easier accurate assessment could made texts survived texts still missing several digitizing projects began india nepal europe north america today decade work digital archives thousands tibetan projects include nitartha international document input center http asian classics input project acip http university virginia tibetan himalayan digital library http tibetan buddhist resource center http otani tibetan project otani university kyoto japan drukpa kagyu heritage project http electronic catalogues tibetan mongolian collections buryatia project institute mongolian tibetan buddhist studies siberian branch russian academy sciences ulan ude buryatia research institute inner asian studies http example one tibetan scholars committed digitizing tibetan canon dzogchen ponlop rinpoche established nitartha international document input center kathmandu nepal team tibetan monks tibetan refugees born diaspora western computer specialists input center indigenous tibetan scholars represent classical tibetan texts digital formats would useful tibetan scholars exile tibetan translators addition digitization texts nitartha international developed tibetan font software several informative websites tibetan buddhism modern educational materials linked classical texts cds interactive modules outline logic philosophical arguments define basic concepts used buddhist philosophy represent key points visual images recently indigenous tibetan scholars also begun learn tei encoding digitized texts text encoding initiative tei xml language designed representing literary historical texts maintained international consortium major universities tei publishes detailed guidelines documentation language maintains listserv questions general discussion amongst tei users holds yearly meeting hundreds tei projects encoders scattered around world tibetan scholars see tei markup language flexible text publishing method digital texts cataloguing schema archiving texts application make easier conduct analytical searches link texts translations one obstacle using tei xml language tibetan canon unicode tibetan language became available recently partial form even though texts encoded unicode printed tibetan unicode script example tibetan scholars experimenting tei ancient texts shows promise tei multicultural texts multiculturalism theoretical approach aims respect autonomy value cultures recently tei consortium initiated project translate guidelines many languages possible without denying concrete economic political forces create digital divide tei consortium intends support indigenous scholars protect sustain textual traditions way text encoding initiative expanding work beyond original north american european borders becoming global digital technology supportive multicultural textual traditions tei consortium welcomed tibetan forays started western scholarly undertaking tei members understand text preservation multicultural especially minority populations important providing indigenous scholars teachers tools keeping textual traditions alive helps secure future literatures world tei important strategy preserving rare texts tei guidelines offer standard set schemas documentation widely used international scholarly community representing texts scholarly framework indigenous tibetan scholars introduce textual tradition wider scholarly community centuries tibetan scholars developed painstaking system marking internal organization texts system delineates several layers scholastic commentaries original sanskrit root text inherent complexity makes tibetan canon natural tei encoding tei guidelines designed complex historical texts mind texts expected endure ages guidelines also offer specific ways represent complicated scholarly features historical texts precision tei encoding good match complexity tibet commentarial tradition tibetan commentaries form nesting structure later commentaries include surround earlier commentaries include surround original indian uncommon tibetan commentary tibetan commentary earlier commentary indian sanskrit root text sections complicated numbering mention incredibly long titles example karmapa mikyö dorje commentary chandrakirti commentary nagarjuna root text includes famous passage explains distinction two truths relative truth ultimate truth passage extensive explanation individual natures two truths numbered title commentary karmapa mikyö dorje chariot dakpo kagyü siddhas quintessential oral instructions glorious düsum khyenpa explaining chandrakirti entrance middle way translated elizabeth callahan included commentary dzogchen ponlop rinpoche entitled commentary chariot dakpo kagyü siddhas sixth mind generation manifest published nitartha institute tibetan commentaries proceed paragraph paragraph verse verse sometimes even line line explicating meaning earlier text complicated internal divisions text captured tei encoding tei encoding also represent linkages texts example original tibetan text translations text western languages example consider following passage tibetan commentary concerns differences two philosophical schools consequentialists autonomists indented verses root text nagarjuna fundamental verses centrism generally tibetan commentaries one quote lines earlier root text explain meaning quoted passage detail explore philosophical debates arisen response passage passage english translation treasury knowledge chapter madhyamaka jamgön kongtrul lodrö thaye translated karl brunnhölzl nitartha institute system centrists following sutras general originated many ways commenting intention text fundamental verses centrism called supreme knowledge noble nagarjuna however mainly fall two schools consequentialists autonomists first verse eulogy beginning nagarjuna treatise reads others without cause place time entities lack arising way explain meaning verse first four claims four extremes master buddhapalita invalidated four antitheses claims set means prove actual thesis master bhavaviveka criticized way buddhapalita formulated invalidation set four initial sentences autonomous probative arguments also proved subject property autonomous way venerable chandrakirti explained bhavaviveka critique buddhapalita apply bhavaviveka acceptance autonomous arguments context explanation reasonings analyze ultimate flawed thus one founded tradition consequentialists extensive manner tei encoding passage head argument system name philosophical school centrists following sutras general originated many ways commenting intention text title italic fundamental verses name philosophical school centrism called supreme knowledge noble persname nagarjuna however mainly fall two schools name philosophical school consequentialists name philosophical school autonomists first verse eulogy beginning persname nagarjuna treatise reads others without cause mdash place time entities lack way explain meaning verse first four claims basic argument four extremes master persname buddhapalita invalidated four antitheses claims set means prove actual thesis master persname bhavaviveka criticized way persname buddhapalita formulated invalidation set four initial sentences basic concept autonomous probative arguments also proved subject property autonomous way venerable persname chandrakirti explained persname bhavaviveka critique persname buddhapalita apply persname bhavaviveka acceptance autonomous arguments context explanation reasonings analyze basic concept ultimate flawed thus one founded tradition name philosophical school consequentialists extensive manner challenge using tei capture tibetan materials lies escapes kind basic encoding tei initially developed scholars familiar western textual tradition features textual traditions may pose theoretical practical challenges may require adaptation tei example one important feature enunciative mode associated text irrelevant tibetan scholars whether text chanted read silently transmitted teacher lung ritual empowers listeners practice meditation study denoted text scanned computer screen tibetans believe texts simply record information actually embody ancient ideas meanings situations require appropriate methods transmission thus understand tibetan texts according theoretical operational paradigm treats information would lose original tibetan cultural context would lose different degrees transmission power said generated different media reading chanting lung transmission etc within tibetan textual tradition representation enunciative mode essential practicing tibetan texts might accomplished additional customized tei elements chanted sections texts another challenging dimension readers actions association text advanced tibetan meditation techniques include hand movements called mudras figure shows examples gestures bodily movements prostrations chanting complex visualizations texts always central meditation methods engagement vajrayana text far active occurs simply reading even chanting text khenpo karthar rinpoche bardor tulku rinpoche leading vajrayana meditation chanting text mudras image two tibetan monks leading meditation example one sections medicine buddha meditation used healing requires special hand movements mudras performed imagining dark blue buddha figure one chants mantra medicine buddha body mudras speech mantra mind visualization blue buddha figure see figure difficult necessary vajrayana meditation practice instructions perform mudras chant mantra tibetan text image blue buddha text either traditionally mudras chant melody must learned tibetan teacher visualization image usually memorized focusing thangka painting shrine room thangka painting medicine buddha karma triyana dharmachakra thangka painting medicine buddha additional dimensions simply contextual information necessary understand text encoded important sense integral parts text meaning without said truly preserved tibetans intent upon preserving textual heritage simply information storage instead aim real preservation records encodes enough living textual tradition texts may taken future generations way continues tibetan culture meaningful values customs transcription simple digitization tibetan texts constitutes mere storage preservation texts may read studied future vajrayana elements mantra mudra meditation captured since elements need practiced live speak read tibetan text sits library digital archive one knows chant meditate debate response text lost virtually meaning fossil text would preserved text would become extinct real preservation requires help indigenous text custodians teach traditional ways working texts pressing concern older tibetan teachers born tibet diaspora die next decade two gone may longer buddhist practitioners know exactly chant certain texts perform mudras connection specific textual passages lead complex rituals based texts preserve complete record possible texts live meditation practices rituals performed texts gestural musical mental dimensions texts must ideally also recorded linking audio video images transcriptions may help may also worth exploring ways notate gestural information within transcription tei offers approaches chapters transcription speech performance texts could extended accommodate distinctive combination information carried tibetan texts encoded tei file also form basis complex multimedia representation tei version medicine buddha liturgy linked audio file correct pronunciation melody mantra pictures hand mudras must performed chanting text visual image blue buddha figure tei transcription also contain instructions perform mudras bodily movements occur point text allowing displayed suppressed appropriate linking tei transcription multimedia resources editor could allow appearance videos graphics photos memory aids certain points text would help perform visualization buddha figure example shown figure gives text chant multiple scripts together description mudras performed mudras illustrated table practitioner voicing chant could corrected accompanied audio file indigenous tibetan chant master articulation audio file tibetan horns symbols bells medicine buddha text heart center karma thegsum chöling medicine buddha text diagram mudras associated medicine buddha text argham padyam pupe dhupe aloke gendhe newidye shapda click audio file audio file tibetan mantra tibetan mantra audio file karma kagyu institute chanted umdze lodro samphel could text encoding initiative help indigenous text custodians around world struggling protect ancient cultural heritages ideally might envision creating simple shared descriptive system facilitate cataloguing endangered texts international scholars become aware texts lost saved ideally system would easy learn scholar working language however significant differences textual traditions descriptive goals make hard create system would simple widely agreed upon may practical work disseminating tei expertise widely among text custodians second step therefore would hold workshops digital technologies universities teachers colleges developing countries digital divide closing parts world example india outreach scholarly communities poorer scholarly communities could include sharing digital methods preserving texts supported occasional workshops virtual international community text custodians trained tei could grow bridge digital divide developed developing countries endeavor important respect control indigenous scholars textual heritage textual heritage cultural property speak world maintained people whose ancestors created traditional rules access certain texts digital technologies bypass rules digital technologies used appropriate world textual riches simply add inventory western digital archives model broad access often motivates western digitization efforts apply universally may cases directly indigenous textual tradition issue comes regard tibetan texts texts esoteric texts reserved advanced meditators generally presumed western scholars increased access texts better presumption shared tibetan scholars deal texts require special permission instruction qualified teacher read studied chanted memorized restricted nature tibetan texts relates difficulty meditation practices described text everyone advanced meditation skills sufficient commitment undertake mental training methods relayed text chinese invasion tibet monastic libraries contained esoteric volumes never circulated beyond monastery even amongst members monastery thus although western scholars consider trespassing serious textual practice crime tibetan scholars might regard texts inviting trespassers essential multicultural development tei involve indigenous scholars understand threats faced endangered textual tradition committed protecting culture endangered tibetan culture members culture try preserve texts try continue rituals practices keep culture alive transcribing texts digitizing important preservation methods methods need"},
 
     ]
```


```python
output_1 = ""
```

The assistant gives you an ouput(#1)

**Conversation_2**

1. Append the conversation history soo far i.e Conversation 1
2. Append the results of the assistant i.e output(#1)
3. Ask GPT to return only the keywords that are semantically related to Data Viz



```python
conversation_2 = [
    conversation_1,
    
    {"role": "assistant", 
    "content": "output_1"},
    
    {"role": "user", 
    "content": '''Return only the keywords that are semantically related to data visualisation in Output 1
          - Return them as comma seperated values.
          - If none exist, return nothing'''}
    
]
```

### Version 3

Chain of thought with few-shot learning followed by prompting with second LLM    
**Conversation_1**

1. You assign a role to the system or the assistant
2. You provide the assistant with a few examples about how to do the task - fewshot learning


```python
conversation = [
    {"role": "system", 
     "content": '''You are an author of a dictionary containing Data visualization keywords. 
    _______________________________
    Your task is to identify Top 10 data visualization related keywords present in the input transcript. 
    ________________________________
    Return them as comma seperated values.'''},
    
    {"role": "user", 
    "content": '''Input 1 : Visualization tools allow humanists make sense of large sets of data in the form of graphs, 
    charts,   infographics, information dashboards, and more. By using quantitative data taken from artifacts 
    such as texts and maps or demographic data such as surveys and census results, humanists can support more 
    traditional types of qualitative research by embedding information visualizations into their writing 
    and presentations. Visualization tools can aid in the discovery of larger patterns related to artifacts 
    that they would not be able to see simply by looking at the data. However, the visualizations can also obscure 
    information or reinforce biases and silences in the data. Therefore, an important component of the 
    digital humanities scholarship is using a critical lens to “close read” information visualizations and 
    the datasets they depict. '''}, 
    
    {"role": "assistant",
    "content": '''Output 1 : visualization, inforgraphics,  graphs, charts, information dashboards, quantitative data, 
    qualitative research, maps, data mapping'''},
    
    {"role": "user", 
     "content": '''Input 2 : Such visual displays, including graphs and charts, may present themselves as objective 
    even unmediated views of reality, rather than a rhetorical constructs. 
    The expression of quantifiable or quantitative information in graphic form, such as bar and pie chart.
    Visualizations of data derived from large-scale data sets such as social networks, digitized corpora, 
    and demographic data.As with so many aspects of digital work, these technologies are intertwined with 
    traditional methods. Knowing what and how to read the visualized forms is at the basis of digital 
    literacy and the assessment of meaning in there new formats.'''},
    
     {"role": "assistant",
    "content": '''Output 2 : graphs, visual, display, charts, bar chart, pie chart, social networks, digitized corpora'''},
    
    {"role": "user", 
    "content": "Input 3 : keeping tradition buddhist philosophy meditation alive long died india century recent efforts digitize materials textual tradition offer opportunities broaden circulation rare materials exiled tibetan scholarly community also suggest conceptual challenges arising complexity texts inherently multimodal character paper describes scholarly meditative traditions texts come discusses possible approaches digitization describes scholarly meditative traditions buddhist sutras india come discusses possible approaches digitization thousand years tibet preserved translated ancient buddhist sutras india keeping tradition buddhist philosophy meditation alive long died india century accuracy tibetan translations sanskrit buddhist texts meant modern scholars could establish oldest versions buddhist theories reliability century scholar max müller noted tibetan version asvaghosha appears much closer original sanskrit chinese fact verbal accuracy often reproduce exact words original since certain sanskrit words always represented tibetan equivalents instance prepositions prefixed verbal roots known care translation also understanding buddhism tibet monastic scholars added commentaries original indian root texts established hundreds libraries universities texts could studied ages tibet ancient texts received careful handling storage sacred texts printed hand wood blocks wrapped silk covers stored shelves behind main altars shrine rooms monasteries temples representing one three jewels buddhism dharma texts instruct buddhists achieve enlightenment canonical texts represent one piece always multimedia enactment key meanings vajrayana buddhism particular form buddhism tibet inherited india vajrayana buddhism also known tantra emphasizes intensive meditation meditation methods incorporate complex visual imagery bodily movements sounds drums horns well chanting texts within overall container quiet meditation practice instructions visualizations bodily gestures chanting practices passed oral lineages teacher student codified sacred rituals monastics joined together learning standardizing practices today tibetan textual tradition faces extinction unless serious efforts made preserve vajrayana practices supports years following chinese invasion tibet tibetan scholars struggled preserve circulate classical texts ancient buddhist heritage many books lost chinese launched cultural revolution tibetan soil destroying libraries temples monasteries universities tibet kept books safe important texts elucidating tibet distinctive vajrayana meditation methods liturgies philosophical investigations may never found although search tibetans still continues today famous commentaries instruction manuals tibetan scholars fled india first attempted preserve texts carried printing texts without delay time paper available india poor quality paper rice paper print bled sides making texts illegible many classical tibetan texts never reprinted left original woodblock form called pecha texts see figure lack facilities poor economic conditions tibetan refugees scattered abroad introduced added risks storage paper texts simply transcribing rescued texts protecting archives stopgap measures tibetan refugees living india europe north america south america parts asia impossible establish centrally located archives tibetans could easily study texts learn highly detailed vajrayana meditation methods described texts western scholars meditators became interested learning tibetan buddhism also found difficult locate understand old tibetan texts mentioned recent commentaries indigenous tibetan scholars unable verify ancient texts actually preserved texts stored iconography complex symbolism associated vajrayana texts explained texts tibetan pecha texts image three tibetan texts late tibetan scholars recognized possibilities digital formats preserving texts circulating texts amongst members tibetan community digitized texts could made accessible tibetan western scholars around world internet digital formats also made cataloguing texts searching specific titles easier accurate assessment could made texts survived texts still missing several digitizing projects began india nepal europe north america today decade work digital archives thousands tibetan projects include nitartha international document input center http asian classics input project acip http university virginia tibetan himalayan digital library http tibetan buddhist resource center http otani tibetan project otani university kyoto japan drukpa kagyu heritage project http electronic catalogues tibetan mongolian collections buryatia project institute mongolian tibetan buddhist studies siberian branch russian academy sciences ulan ude buryatia research institute inner asian studies http example one tibetan scholars committed digitizing tibetan canon dzogchen ponlop rinpoche established nitartha international document input center kathmandu nepal team tibetan monks tibetan refugees born diaspora western computer specialists input center indigenous tibetan scholars represent classical tibetan texts digital formats would useful tibetan scholars exile tibetan translators addition digitization texts nitartha international developed tibetan font software several informative websites tibetan buddhism modern educational materials linked classical texts cds interactive modules outline logic philosophical arguments define basic concepts used buddhist philosophy represent key points visual images recently indigenous tibetan scholars also begun learn tei encoding digitized texts text encoding initiative tei xml language designed representing literary historical texts maintained international consortium major universities tei publishes detailed guidelines documentation language maintains listserv questions general discussion amongst tei users holds yearly meeting hundreds tei projects encoders scattered around world tibetan scholars see tei markup language flexible text publishing method digital texts cataloguing schema archiving texts application make easier conduct analytical searches link texts translations one obstacle using tei xml language tibetan canon unicode tibetan language became available recently partial form even though texts encoded unicode printed tibetan unicode script example tibetan scholars experimenting tei ancient texts shows promise tei multicultural texts multiculturalism theoretical approach aims respect autonomy value cultures recently tei consortium initiated project translate guidelines many languages possible without denying concrete economic political forces create digital divide tei consortium intends support indigenous scholars protect sustain textual traditions way text encoding initiative expanding work beyond original north american european borders becoming global digital technology supportive multicultural textual traditions tei consortium welcomed tibetan forays started western scholarly undertaking tei members understand text preservation multicultural especially minority populations important providing indigenous scholars teachers tools keeping textual traditions alive helps secure future literatures world tei important strategy preserving rare texts tei guidelines offer standard set schemas documentation widely used international scholarly community representing texts scholarly framework indigenous tibetan scholars introduce textual tradition wider scholarly community centuries tibetan scholars developed painstaking system marking internal organization texts system delineates several layers scholastic commentaries original sanskrit root text inherent complexity makes tibetan canon natural tei encoding tei guidelines designed complex historical texts mind texts expected endure ages guidelines also offer specific ways represent complicated scholarly features historical texts precision tei encoding good match complexity tibet commentarial tradition tibetan commentaries form nesting structure later commentaries include surround earlier commentaries include surround original indian uncommon tibetan commentary tibetan commentary earlier commentary indian sanskrit root text sections complicated numbering mention incredibly long titles example karmapa mikyö dorje commentary chandrakirti commentary nagarjuna root text includes famous passage explains distinction two truths relative truth ultimate truth passage extensive explanation individual natures two truths numbered title commentary karmapa mikyö dorje chariot dakpo kagyü siddhas quintessential oral instructions glorious düsum khyenpa explaining chandrakirti entrance middle way translated elizabeth callahan included commentary dzogchen ponlop rinpoche entitled commentary chariot dakpo kagyü siddhas sixth mind generation manifest published nitartha institute tibetan commentaries proceed paragraph paragraph verse verse sometimes even line line explicating meaning earlier text complicated internal divisions text captured tei encoding tei encoding also represent linkages texts example original tibetan text translations text western languages example consider following passage tibetan commentary concerns differences two philosophical schools consequentialists autonomists indented verses root text nagarjuna fundamental verses centrism generally tibetan commentaries one quote lines earlier root text explain meaning quoted passage detail explore philosophical debates arisen response passage passage english translation treasury knowledge chapter madhyamaka jamgön kongtrul lodrö thaye translated karl brunnhölzl nitartha institute system centrists following sutras general originated many ways commenting intention text fundamental verses centrism called supreme knowledge noble nagarjuna however mainly fall two schools consequentialists autonomists first verse eulogy beginning nagarjuna treatise reads others without cause place time entities lack arising way explain meaning verse first four claims four extremes master buddhapalita invalidated four antitheses claims set means prove actual thesis master bhavaviveka criticized way buddhapalita formulated invalidation set four initial sentences autonomous probative arguments also proved subject property autonomous way venerable chandrakirti explained bhavaviveka critique buddhapalita apply bhavaviveka acceptance autonomous arguments context explanation reasonings analyze ultimate flawed thus one founded tradition consequentialists extensive manner tei encoding passage head argument system name philosophical school centrists following sutras general originated many ways commenting intention text title italic fundamental verses name philosophical school centrism called supreme knowledge noble persname nagarjuna however mainly fall two schools name philosophical school consequentialists name philosophical school autonomists first verse eulogy beginning persname nagarjuna treatise reads others without cause mdash place time entities lack way explain meaning verse first four claims basic argument four extremes master persname buddhapalita invalidated four antitheses claims set means prove actual thesis master persname bhavaviveka criticized way persname buddhapalita formulated invalidation set four initial sentences basic concept autonomous probative arguments also proved subject property autonomous way venerable persname chandrakirti explained persname bhavaviveka critique persname buddhapalita apply persname bhavaviveka acceptance autonomous arguments context explanation reasonings analyze basic concept ultimate flawed thus one founded tradition name philosophical school consequentialists extensive manner challenge using tei capture tibetan materials lies escapes kind basic encoding tei initially developed scholars familiar western textual tradition features textual traditions may pose theoretical practical challenges may require adaptation tei example one important feature enunciative mode associated text irrelevant tibetan scholars whether text chanted read silently transmitted teacher lung ritual empowers listeners practice meditation study denoted text scanned computer screen tibetans believe texts simply record information actually embody ancient ideas meanings situations require appropriate methods transmission thus understand tibetan texts according theoretical operational paradigm treats information would lose original tibetan cultural context would lose different degrees transmission power said generated different media reading chanting lung transmission etc within tibetan textual tradition representation enunciative mode essential practicing tibetan texts might accomplished additional customized tei elements chanted sections texts another challenging dimension readers actions association text advanced tibetan meditation techniques include hand movements called mudras figure shows examples gestures bodily movements prostrations chanting complex visualizations texts always central meditation methods engagement vajrayana text far active occurs simply reading even chanting text khenpo karthar rinpoche bardor tulku rinpoche leading vajrayana meditation chanting text mudras image two tibetan monks leading meditation example one sections medicine buddha meditation used healing requires special hand movements mudras performed imagining dark blue buddha figure one chants mantra medicine buddha body mudras speech mantra mind visualization blue buddha figure see figure difficult necessary vajrayana meditation practice instructions perform mudras chant mantra tibetan text image blue buddha text either traditionally mudras chant melody must learned tibetan teacher visualization image usually memorized focusing thangka painting shrine room thangka painting medicine buddha karma triyana dharmachakra thangka painting medicine buddha additional dimensions simply contextual information necessary understand text encoded important sense integral parts text meaning without said truly preserved tibetans intent upon preserving textual heritage simply information storage instead aim real preservation records encodes enough living textual tradition texts may taken future generations way continues tibetan culture meaningful values customs transcription simple digitization tibetan texts constitutes mere storage preservation texts may read studied future vajrayana elements mantra mudra meditation captured since elements need practiced live speak read tibetan text sits library digital archive one knows chant meditate debate response text lost virtually meaning fossil text would preserved text would become extinct real preservation requires help indigenous text custodians teach traditional ways working texts pressing concern older tibetan teachers born tibet diaspora die next decade two gone may longer buddhist practitioners know exactly chant certain texts perform mudras connection specific textual passages lead complex rituals based texts preserve complete record possible texts live meditation practices rituals performed texts gestural musical mental dimensions texts must ideally also recorded linking audio video images transcriptions may help may also worth exploring ways notate gestural information within transcription tei offers approaches chapters transcription speech performance texts could extended accommodate distinctive combination information carried tibetan texts encoded tei file also form basis complex multimedia representation tei version medicine buddha liturgy linked audio file correct pronunciation melody mantra pictures hand mudras must performed chanting text visual image blue buddha figure tei transcription also contain instructions perform mudras bodily movements occur point text allowing displayed suppressed appropriate linking tei transcription multimedia resources editor could allow appearance videos graphics photos memory aids certain points text would help perform visualization buddha figure example shown figure gives text chant multiple scripts together description mudras performed mudras illustrated table practitioner voicing chant could corrected accompanied audio file indigenous tibetan chant master articulation audio file tibetan horns symbols bells medicine buddha text heart center karma thegsum chöling medicine buddha text diagram mudras associated medicine buddha text argham padyam pupe dhupe aloke gendhe newidye shapda click audio file audio file tibetan mantra tibetan mantra audio file karma kagyu institute chanted umdze lodro samphel could text encoding initiative help indigenous text custodians around world struggling protect ancient cultural heritages ideally might envision creating simple shared descriptive system facilitate cataloguing endangered texts international scholars become aware texts lost saved ideally system would easy learn scholar working language however significant differences textual traditions descriptive goals make hard create system would simple widely agreed upon may practical work disseminating tei expertise widely among text custodians second step therefore would hold workshops digital technologies universities teachers colleges developing countries digital divide closing parts world example india outreach scholarly communities poorer scholarly communities could include sharing digital methods preserving texts supported occasional workshops virtual international community text custodians trained tei could grow bridge digital divide developed developing countries endeavor important respect control indigenous scholars textual heritage textual heritage cultural property speak world maintained people whose ancestors created traditional rules access certain texts digital technologies bypass rules digital technologies used appropriate world textual riches simply add inventory western digital archives model broad access often motivates western digitization efforts apply universally may cases directly indigenous textual tradition issue comes regard tibetan texts texts esoteric texts reserved advanced meditators generally presumed western scholars increased access texts better presumption shared tibetan scholars deal texts require special permission instruction qualified teacher read studied chanted memorized restricted nature tibetan texts relates difficulty meditation practices described text everyone advanced meditation skills sufficient commitment undertake mental training methods relayed text chinese invasion tibet monastic libraries contained esoteric volumes never circulated beyond monastery even amongst members monastery thus although western scholars consider trespassing serious textual practice crime tibetan scholars might regard texts inviting trespassers essential multicultural development tei involve indigenous scholars understand threats faced endangered textual tradition committed protecting culture endangered tibetan culture members culture try preserve texts try continue rituals practices keep culture alive transcribing texts digitizing important preservation methods methods need"},
     ]
```


```python
output_1 = ""
```

The assistant gives you an ouput(#1)

**Conversation_2**

1. Define a new LLM with a task to evaluate the keywords generated by the previous LLM
2. Append the output of the previous assistant i.e output(#1) to the conversation
3. You can use a more powerful LLM in this use case as the number of tokens will be much lesser



```python
# Use a new LLM to run this conversation

conversation_2 = [
    {"role": "system", 
     "content": '''You take input from another LLM that performs keyword recognition. 
     
     It's output is generic. 
     
     I want you to only return the keywords that are semantically related to data visualization. 
     
     All the keywords should be present in the user input.
     
     Return them as comma seperated values.'''},
    
    {"role": "user", 
    "content": output_1}, 
    

]
```

#### Pros and Cons

Pros and Cons of all 3 versions were discussed. 

Here are a few notes:

- Version 1 is too vanilla, but it is a good start to test and debug results
- Version 2 is powerful but the conversation history becomes  too large for the LLM to process, since we are doing the same task twice
- Version 3 was the method we agreed to take forward. Using another LLM to evaluate the results of a previous conversation is the train of thought which made most sense

#### Next Steps

- Score each keyword that the previous LLM provides, this score can be used to filter relevant keywords.
- How does LLM make the evalutation? 
- Is LLM using an algorithm to make an evalutation?
- Enhance the results of Version 3 and run it on the whole corpus.

# Phase 6


#### Task 1 : Come up with a signifiance scoring for each keyword generated by LLM



```python
conversation = [
    {"role": "system", 
     "content": '''You take input from another LLM that performs keyword recognition. 
     
    Your task is to score all words between a scale of 0 and 1. The score for each word
    is determined by how semantically relevant it is with respect to data visualization. 
          
    Return the results in descending order'''},
    
    {"role": "user", 
    "content": "Input from user"}, 
    
]
```

This framework allows us to define a scoring mechanism for the keywords generated by the LLM. But how does GPT come up with this scoring?

#### Task 2 : Identify how the LLM comes up with this scoring


```python
conversation = [
    {"role": "system", 
     "content": '''You take input from another LLM that performs keyword recognition. 
     
    Your task is to score all words between a scale of 0 and 1. The score for each word
    is determined by how semantically relevant it is with respect to data visualization. 
          
    '''},
    
    {"role": "user", 
    "content": "Input from user"}, 
        
    {"role": "assistant", 
    "content": "Output from assistant"}, 
    
    {"role": "user", 
    "content": '''How did you come up with this scoring? 
     Can you tell me what algorithm or methodology you followed.
     What steps did you follow to idenitfy these words'''}
]
```

GPT gives a different response everytime you ask it the same question. Most questions had a similar pattern that GPT follows:


**From the responses alone - we can understand how it generates its output.**    
''' As an AI language model, I don't have a formula or algorithm specifically designed for word scoring. Instead, I base my judgment on my training data, which contains a broad cross-section of human language. My answers are generated based on patterns and information I've learned from these data. '''    

- GPT is a text-competion model
- Based on the probability of the next occuring word, it completes the sentence
- Does not perform any algorithm under the hood

#### Task 3: Run on corpus

Upon attempting to run a chain-of-thought prompt style conversation for the whole corpus there were some limitations.

- GPT threw errors regarding the rate limits
- The number of tokens you can parse through GPT is restricted based on time
- We need to EFFICIENTLY prompt GPT, and not prompt it only solely based on the what is the RIGHT way


Based on the learnings from all the experiments, here is the base LLM chain that we arrived at


```python
LLM_1 = """ You are an author of a dictionary containing Data Visualization keywords. 
            _______________________________
            Your task is to identify the top 10 keywords related to Data Visualization that are present in the input transcript.
            ________________________________
            Return the keywords as comma seperated values without any numbering.

            Transcript: {question}
        """
    
    
LLM_2 = """ You take input from another LLM that performs keyword recognition. 

        Your task is to score all words between a scale of 0 and 1. The score for each word
        is determined by how semantically related it is with respect to data visualization. 

        Return in the format - [[keyword, score]]
        {list_keys}

        """
```

# Phase 7


Prompt to identify the keywords - GPT 3.5


```python
prompt_template = """ You are an author of a dictionary containing Data Visualization keywords. 
    _______________________________
    Your task is to identify the top 10 keywords related to Data Visualization that are present in the input transcript.
    ________________________________
    Return the keywords as comma seperated values without any numbering.
    
    Transcript: {question}
"""
```

Prompt to score keywords based on their significance


```python
prompt_template = """ You take input from another LLM that performs keyword recognition. 

Your task is to score all words between a scale of 0 and 1. The score for each word
is determined by how semantically related it is with respect to data visualization. 

Return in the format - [[keyword, score]]
{list_keys}

"""
```

### Some Ideas

**Are there data visualization keywords present in the journal?**    
If there are very few or none, then we cannot measure LLM performance.

**How do we classify whether or not it is a keyword or not?**    
How do we subjectively classify whether it is right or wrong. When is a keyword generic, when is it not.

**Manual inspection v LLM**    
If the above are defined, we can compare what we think is a keyword vs what the LLM is saying .

**Fine-tuning GPT**    
Nice video - https://www.youtube.com/watch?v=W4Q9bKLNYiQ&t=947s&ab_channel=AllAboutAI    https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset

- Generate 20/30/X few-shot examples, labeled examples
- Provide trained examples to GPT 3.5 and train it - it costs $
- Use this fine-tuned model to perform the task

#### Current

Manual inspection of the keywords, based on the file generated by GPT.


### Final Prompt

The prompt templates to perform NER finally were the following:


```python
# Prompt template 1
# GPT 3.5
# Identification of keywords from abstarcts of journals

prompt_template = """ You are an author of a dictionary containing Data Visualization keywords. 
_______________________________
Your task is to identify only the top 10 keywords related to Data Visualization that are present in the input transcript.
________________________________
Return the keywords as comma seperated values without any numbering.

Transcript: {question}
"""
```


```python
# Prompt template 2
# GPT 4
# Significance scoring of keywords generated by previous LLM

prompt_template = """ You take input from another LLM that performs keyword recognition. 

Your task is to score all the words between a scale of 0 and 1. The score for each word
is determined by how semantically related it is with respect to data visualization. 

Return in the format - [[keyword, score]]
{list_keys}

"""
```

# Results

After performing a thorough analysis of the different ways to prompt, I ran the analysis on the articles belonging to 3 different journals.

Keywords were generated at a high accuracy and efficiency with the usage of these prompts for the use-case.


```python
!jupyter nbconvert --execute --to markdown "/Users/amar96/Desktop/Ford/PromptEngineering.ipynb"
```

    [NbConvertApp] Converting notebook /Users/amar96/Desktop/Ford/PromptEngineering.ipynb to markdown
    [NbConvertApp] Writing 64984 bytes to /Users/amar96/Desktop/Ford/PromptEngineering.md



```python

```
