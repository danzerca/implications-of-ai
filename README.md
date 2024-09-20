# Midterm

## Background and Context
The CEO and corporate, with permission of the board, have assembled a crack data science and engineering team to take advantage of RAG, agents, and all of the latest open-source technologies emerging in the industry.  This time it's for real though.  This time, the company is aiming squarely at some Return On Investment - some ROI - on its research and development dollars.

## The Problem
You are an AI Solutions Engineer.  You've worked directly with internal stakeholders to identify a problem: `people are concerned about the implications of AI, and no one seems to understand the right way to think about building ethical and useful AI applications for enterprises.`

This is a big problem and one that is rapidly changing.  Several people you interviewed said that *they could benefit from a chatbot that helped them understand how the AI industry is evolving, especially as it relates to politics.*  Many are interested due to the current election cycle, but others feel that some of the best guidance is likely to come from the government.

## Task 1: Dealing with the Data
You identify the following important documents that, if used for context, you believe will help people understand what’s happening now:
1. 2022: [Blueprint for an AI Bill of Rights: Making Automated Systems Work for the American People](https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf) (PDF)
2. 2024: [National Institute of Standards and Technology (NIST) Artificial Intelligent Risk Management Framework](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) (PDF)
Your boss, the SVP of Technology, green-lighted this project to drive the adoption of AI throughout the enterprise.  It will be a nice showpiece for the upcoming conference and the big AI initiative announcement the CEO is planning.

> Task 1: Review the two PDFs and decide how best to chunk up the data with a single strategy to optimally answer the variety of questions you expect to receive from people.
>
> *Hint: Create a list of potential questions that people are likely to ask!*

✅ Deliverables:
1. Describe the default chunking strategy that you will use.
2. Articulate a chunking strategy that you would also like to test out.
3. Describe how and why you made these decisions

## Task 2: Building a Quick End-to-End Prototype
**You are an AI Systems Engineer.**  The SVP of Technology has tasked you with spinning up a quick RAG prototype for answering questions that internal stakeholders have about AI, using the data provided in Task 1.

> Task 2: Build an end-to-end RAG application using an industry-standard open-source stack and your choice of commercial off-the-shelf models

✅ Deliverables:
1. Build a prototype and deploy to a Hugging Face Space, and create a short (< 2 min) loom video demonstrating some initial testing inputs and outputs.
2. How did you choose your stack, and why did you select each tool the way you did?

## Task 3: Creating a Golden Test Data Set
**You are an AI Evaluation & Performance Engineer.**  The AI Systems Engineer who built the initial RAG system has asked for your help and expertise in creating a "Golden Data Set."

> Task 3: Generate a synthetic test data set and baseline an initial evaluation

✅ Deliverables:
1. Assess your pipeline using the RAGAS framework including key metrics faithfulness, answer relevancy, context precision, and context recall.  Provide a table of your output results.

<div>
<style scoped>
.dataframe tbody tr th:only-of-type {
    vertical-align: middle;
}

.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>question</th>
      <th>contexts</th>
      <th>answer</th>
      <th>ground_truth</th>
      <th>faithfulness</th>
      <th>answer_relevancy</th>
      <th>context_recall</th>
      <th>context_precision</th>
      <th>answer_correctness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>How should automated systems be designed to en...</td>
      <td>[You should be able to opt out, where appropri...</td>
      <td>Automated systems should be designed with mech...</td>
      <td>Automated systems should be designed to ensure...</td>
      <td>1.000000</td>
      <td>0.942021</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.678383</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What are the primary information security risk...</td>
      <td>[10 \nGAI systems can ease the unintentional p...</td>
      <td>The primary information security risks associa...</td>
      <td>The primary information security risks associa...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.889574</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What is the role of the National Institute of ...</td>
      <td>[About AI at NIST: The National Institute of S...</td>
      <td>The National Institute of Standards and Techno...</td>
      <td>The National Institute of Standards and Techno...</td>
      <td>1.000000</td>
      <td>0.867493</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.983044</td>
    </tr>
    <tr>
      <th>3</th>
      <td>How can structured feedback about content prov...</td>
      <td>[51 \ngeneral public participants. For example...</td>
      <td>Structured feedback about content provenance c...</td>
      <td>Structured feedback about content provenance c...</td>
      <td>1.000000</td>
      <td>0.988002</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.568428</td>
    </tr>
    <tr>
      <th>4</th>
      <td>How are feedback mechanisms utilized to verify...</td>
      <td>[20 \nGV-4.3-003 \nVerify information sharing ...</td>
      <td>Feedback mechanisms are utilized to verify inf...</td>
      <td>Feedback mechanisms are utilized to verify inf...</td>
      <td>0.818182</td>
      <td>0.939008</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.782742</td>
    </tr>
    <tr>
      <th>5</th>
      <td>What concerns can be mitigated by assessing th...</td>
      <td>[37 \nMS-2.11-005 \nAssess the proportion of s...</td>
      <td>Assessing the proportion of synthetic to non-s...</td>
      <td>Assessing the proportion of synthetic to non-s...</td>
      <td>0.583333</td>
      <td>0.957169</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.456097</td>
    </tr>
    <tr>
      <th>6</th>
      <td>How can evaluations involving human subjects i...</td>
      <td>[30 \nMEASURE 2.2: Evaluations involving human...</td>
      <td>Evaluations involving human subjects in GAI ap...</td>
      <td>Evaluations involving human subjects in GAI ap...</td>
      <td>1.000000</td>
      <td>0.953573</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.993832</td>
    </tr>
    <tr>
      <th>7</th>
      <td>How should national security risks be addresse...</td>
      <td>[17 \nGOVERN 1.7: Processes and procedures are...</td>
      <td>National security risks should be addressed wh...</td>
      <td>When mapping, measuring, and managing AI risks...</td>
      <td>1.000000</td>
      <td>0.978501</td>
      <td>0.5</td>
      <td>1.000000</td>
      <td>0.614079</td>
    </tr>
    <tr>
      <th>8</th>
      <td>What resources are provided by the National In...</td>
      <td>[NIST Trustworthy and Responsible AI  \nNIST A...</td>
      <td>The National Institute of Standards and Techno...</td>
      <td>The National Institute of Standards and Techno...</td>
      <td>1.000000</td>
      <td>0.926981</td>
      <td>0.0</td>
      <td>0.416667</td>
      <td>0.279565</td>
    </tr>
    <tr>
      <th>9</th>
      <td>How can structured feedback mechanisms be used...</td>
      <td>[49 \nearly lifecycle TEVV approaches are deve...</td>
      <td>Structured feedback mechanisms can be used to ...</td>
      <td>Structured feedback mechanisms can be used to ...</td>
      <td>1.000000</td>
      <td>0.967218</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.443375</td>
    </tr>
    <tr>
      <th>10</th>
      <td>How does the AI Bill of Rights protect civil r...</td>
      <td>[APPENDIX\nExamples of Automated Systems \nThe...</td>
      <td>The AI Bill of Rights protects civil rights in...</td>
      <td>The Blueprint for an AI Bill of Rights is inte...</td>
      <td>1.000000</td>
      <td>0.969873</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.417014</td>
    </tr>
    <tr>
      <th>11</th>
      <td>How can proxies in automated systems cause alg...</td>
      <td>[­­­­­­­\nALGORITHMIC DISCRIMINATION Protectio...</td>
      <td>Proxies in automated systems can cause algorit...</td>
      <td>Proxies in automated systems can cause algorit...</td>
      <td>0.904762</td>
      <td>0.911332</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.842794</td>
    </tr>
    <tr>
      <th>12</th>
      <td>How can organizations manage national security...</td>
      <td>[SAFE AND EFFECTIVE \nSYSTEMS \nHOW THESE PRIN...</td>
      <td>Organizations can manage national security ris...</td>
      <td>The answer to given question is not present in...</td>
      <td>0.961538</td>
      <td>0.968402</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.172271</td>
    </tr>
    <tr>
      <th>13</th>
      <td>How do companies use surveillance software to ...</td>
      <td>[DATA PRIVACY \nEXTRA PROTECTIONS FOR DATA REL...</td>
      <td>Companies use surveillance software to track e...</td>
      <td>Companies use surveillance software to track e...</td>
      <td>1.000000</td>
      <td>0.945468</td>
      <td>0.5</td>
      <td>1.000000</td>
      <td>0.617259</td>
    </tr>
    <tr>
      <th>14</th>
      <td>What steps ensure safe monitoring of automated...</td>
      <td>[SAFE AND EFFECTIVE \nSYSTEMS \nWHAT SHOULD BE...</td>
      <td>To ensure safe monitoring of automated systems...</td>
      <td>Automated systems should have ongoing monitori...</td>
      <td>1.000000</td>
      <td>0.959679</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.411017</td>
    </tr>
    <tr>
      <th>15</th>
      <td>How do human-AI systems improve customer servi...</td>
      <td>[HUMAN ALTERNATIVES, \nCONSIDERATION, AND \nFA...</td>
      <td>Human-AI systems improve customer service and ...</td>
      <td>Human-AI systems improve customer service by i...</td>
      <td>0.925000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.283134</td>
    </tr>
    <tr>
      <th>16</th>
      <td>How can stakeholders maintain privacy, transpa...</td>
      <td>[DATA PRIVACY \nWHAT SHOULD BE EXPECTED OF AUT...</td>
      <td>Stakeholders can maintain privacy, transparenc...</td>
      <td>The answer to given question is not present in...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.333333</td>
      <td>0.175027</td>
    </tr>
    <tr>
      <th>17</th>
      <td>How can automated systems be made safe and eff...</td>
      <td>[SAFE AND EFFECTIVE \nSYSTEMS \nWHAT SHOULD BE...</td>
      <td>To make automated systems safe and effective, ...</td>
      <td>The expectations for automated systems are mea...</td>
      <td>1.000000</td>
      <td>0.946864</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.660495</td>
    </tr>
    <tr>
      <th>18</th>
      <td>How can GAI red-teaming be used to monitor and...</td>
      <td>[50 \nParticipatory Engagement Methods \nOn an...</td>
      <td>GAI red-teaming can be utilized to monitor and...</td>
      <td>GAI red-teaming can be used to monitor and imp...</td>
      <td>0.882353</td>
      <td>0.957436</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.600873</td>
    </tr>
  </tbody>
</table>
</div>

2. What conclusions can you draw about performance and effectiveness of your pipeline with this information?

## Task 4: Fine-Tuning Open-Source Embeddings
**You are an Machine Learning Engineer.**  The AI Evaluation and Performance Engineer has asked for your help in fine-tuning the embedding model used in their recent RAG application build.

> Task 4: Generate synthetic fine-tuning data and complete fine-tuning of the open-source embedding model

✅ Deliverables:
1. Swap out your existing embedding model for the new fine-tuned version.  Provide a link to your fine-tuned embedding model on the Hugging Face Hub.
2. How did you choose the embedding model for this application?

{'faithfulness': 0.9513, 'answer_relevancy': 0.9568, 'context_recall': 0.8947, 'context_precision': 0.9342, 'answer_correctness': 0.5721}

{'faithfulness': 0.9379, 'answer_relevancy': 0.9543, 'context_recall': 0.8947, 'context_precision': 0.9342, 'answer_correctness': 0.5718}


## Task 5: Assessing Performance
**You are the AI Evaluation & Performance Engineer.**  It's time to assess all options for this product.

> Task 5: Assess the performance of 1) the fine-tuned model, and 2) the two proposed chunking strategies

✅ Deliverables:
1. Test the fine-tuned embedding model using the RAGAS frameworks to quantify any improvements.  Provide results in a table.
2. Test the two chunking strategies using the RAGAS frameworks to quantify any improvements. Provide results in a table.
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Baseline</th>
      <th>Recursive</th>
      <th>Baseline -&gt; Recursive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>faithfulness</td>
      <td>0.938889</td>
      <td>0.832334</td>
      <td>-0.106555</td>
    </tr>
    <tr>
      <th>1</th>
      <td>answer_relevancy</td>
      <td>0.869088</td>
      <td>0.813999</td>
      <td>-0.055089</td>
    </tr>
    <tr>
      <th>2</th>
      <td>context_recall</td>
      <td>0.888889</td>
      <td>0.885965</td>
      <td>-0.002924</td>
    </tr>
    <tr>
      <th>3</th>
      <td>context_precision</td>
      <td>0.870370</td>
      <td>0.785088</td>
      <td>-0.085283</td>
    </tr>
    <tr>
      <th>4</th>
      <td>answer_correctness</td>
      <td>0.658931</td>
      <td>0.680589</td>
      <td>0.021658</td>
    </tr>
  </tbody>
</table>
</div>

Training model:
{'train_runtime': 89.5115, 'train_samples_per_second': 9.183, 'train_steps_per_second': 0.603, 'train_loss': 0.7247594904016565, 'epoch': 3.0}

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Baseline</th>
      <th>Fine-tuned</th>
      <th>Baseline -&gt; Fine-tuned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>faithfulness</td>
      <td>0.933532</td>
      <td>0.055556</td>
      <td>-0.877976</td>
    </tr>
    <tr>
      <th>1</th>
      <td>answer_relevancy</td>
      <td>0.864565</td>
      <td>0.000000</td>
      <td>-0.864565</td>
    </tr>
    <tr>
      <th>2</th>
      <td>context_recall</td>
      <td>0.888889</td>
      <td>0.147222</td>
      <td>-0.741667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>context_precision</td>
      <td>0.870370</td>
      <td>0.510802</td>
      <td>-0.359568</td>
    </tr>
    <tr>
      <th>4</th>
      <td>answer_correctness</td>
      <td>0.646531</td>
      <td>0.215933</td>
      <td>-0.430599</td>
    </tr>
  </tbody>
</table>
</div>


3. The AI Solutions Engineer asks you “Which one is the best to test with internal stakeholders next week, and why?”

## Task 6: Managing Your Boss and User Expectations
**You are the SVP of Technology.**  Given the work done by your team so far, you're now sitting down with the AI Solutions Engineer.  You have tasked the solutions engineer to test out the new application with at least 50 different internal stakeholders over the next month.
1. What is the story that you will give to the CEO to tell the whole company at the launch next month?
2. There appears to be important information not included in our build, for instance, the [270-day update](https://www.whitehouse.gov/briefing-room/statements-releases/2024/07/26/fact-sheet-biden-harris-administration-announces-new-ai-actions-and-receives-additional-major-voluntary-commitment-on-ai/) on the 2023 executive order on [Safe, Secure, and Trustworthy AI](https://www.whitehouse.gov/briefing-room/presidential-actions/2023/10/30/executive-order-on-the-safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence/).  How might you incorporate relevant white-house briefing information into future versions?

## Your Final Submission
Please include the following in your final submission:
1. A public link to a **written report** addressing each deliverable and answering each question.
2. A public link to any relevant **GitHub repo**
3. A public link to the **final version of your application** on Hugging Face

4. A public link to your **fine-tuned embedding model** on Hugging Face
https://huggingface.co/danicafisher/dfisher-sentence-transformer-fine-tuned