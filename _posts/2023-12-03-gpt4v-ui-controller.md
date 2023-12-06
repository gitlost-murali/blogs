---
title: Creating an Automated UI controller with GPT agents & GPT-4v
excerpt: Talks about how we can build an automated UI controller with GPT agents
tags: [Machine Learning, Language Models, GPT, LLM, Product Development, Prototyping, huggingface, transformers, OpenAI, GPT4, GPT3, agents, vision, ui, automation]
date: 2023-12-03 05:28:10 +0530
categories: machine-learning product-development gpt4v automation
toc: true
permalink: /:categories/:title
---

# 1. Background
The advent of LLMs and tools like ChatGPT marked a new era in text generation and AI progress. There have been attempts like AutoGPT to achieve task automation. However, the problem with such text-LLM based tools is their reliance on underlying DOM/HTML code. Since most desktop applications are based on .NET/.SAP, it would be difficult to convert the visual into a text format and feed it to the LLMs. The latest release of GPT-4 vision (GPT-4v) API offers a breakthrough, bypassing the need for HTML code reliance and focusing instead on visual inputs.

## The Challenge of Grounding

Now that GPT4-Vision (GPT-4v) is available as an API, it is possible to abstain HTML code and just focus on the visual input. Specifically, it has become as straightforward as taking a screenshot of the current state and asking GPT4-V which action to perform (like click on the next button, double click, etc).

Despite GPT-4v’s prowess in image analysis, its precision in identifying UI element locations remains a hurdle. For instance, in the image, it struggles to output the correct bounding boxes or locations of the people in the image.

<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/vision-agents/gpt4v-bboxes.png"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/vision-agents/gpt4v-bboxes.png"></a>
    <figcaption><b>Figure 1:</b> GPT-4 Vision model struggling to locate people correctly. (Source: https://huyenchip.com/2023/10/10/multimodal.html) </figcaption>
</figure>

# Solution: Grounding

__Grounding__, the ability to accurately pinpoint these locations, is crucial. Strategies like SoM (Set of Mark) enhance grounding by pre-annotating the image with tools like Segment-Anything (SAM) where the objects in the image are segmented and numbered. This process simplifies the process for GPT-4v to identify the correct object by its numbered segmentation mask.

<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/vision-agents/som.webp"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/vision-agents/som.webp"></a>
    <figcaption><b>Figure 2:</b> Set of Mark example </figcaption>
</figure>

## Adapting to the usecase
In the context of UI images, the elements can be numbered as below. Let’s say if “Start for Free” button has to be clicked for next steps, now GPT-4v can just mention the number “34”. Later, we connect the number “34” with the bounding boxes predicted by existing object detection tool to move mouse the corresponding coordinates and perform controller actions.

<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/vision-agents/som_ui.webp"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/vision-agents/som_ui.webp"></a>
    <figcaption><b>Figure 3:</b> Object Detection on a UI screen </figcaption>
</figure>

# 2. Towards building Agents

Based on this idea, we want to build a product/tool where we can enter a goal/task and expect a bunch of things to do it for us.

Let us approach the system, sequentially, in the order it is supposed to work. One capability we need is to breakdown the ultimate goal and plan what steps are needed to achieve that goal. Second capability is vision i.e. image understanding to predict the desired action like click, move mouse, etc. Third capability is to control the system — mimicing user actions like click, double click, move mouse, etc.

So, all we need is one GPT-4v and equip it with controller to interact with the UI environment. However, functions are not supported for GPT-4V, yet. So, an alternative is to have a GPT-4 text model as the orchestrator/agent which can see through GPT-4V and perform actions through the controller.

<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/vision-agents/gpt-controller.jpg"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/vision-agents/gpt-controller.jpg"></a>
    <figcaption><b>Figure 4:</b> UI controller agent which can plan and analyze images </figcaption>
</figure>

# 3. GPT-4V integration: Visionary Function

GPT-4v serves as the visual interpreter in our system. It analyzes screenshots of the UI, identifying and numerating elements for interaction. Below is an example function to request UI element identification:

```python
def request_ui_element_gpt4v(base64_image, query, openai_api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "temperature": 0.1,
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": f"Hey, imagine that you are guiding me navigating the 
                        UI elements in the provided image(s). 
                        All the UI elements are numbered for reference.
                        The associated numbers are on top left of 
                        corresponding bbox. For the prompt/query asked, 
                        return the number associated to the target element 
                        to perform the action. query: {query}"
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            ]
        }
        ],
        "max_tokens": 400
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response.json()
```

# 4. Device Controller: The Interactive Core

The Controller class is pivotal, executing actions within the UI. It includes functions for mouse movements, clicks, text entry, and taking screenshots. Key functions include:

* `move_mouse()`: Moves the cursor and performs clicks.
* `double_click_at_location()`: Executes double clicks at specified locations.
* `enter_text_at_location()`: Inputs text at a given location.
* `take_screenshot()`: Captures and annotates the current UI state.

This class ensures seamless interaction with the system, acting based on GPT-4v's guidance.

```python
import subprocess
import time

class Controller:
    def __init__(self, window_name="Mozilla Firefox") -> None:    
        # Get the window ID (replace 'Window_Name' with your window's title)
        self.window_name = window_name
        self.get_window_id = ["xdotool", "search", "--name", self.window_name]
        self.window_id = subprocess.check_output(self.get_window_id).strip()

    def move_mouse(self, x, y, click=1):
        # AI logic to determine the action (not shown)
        action = {"x": x, "y": y, "click": click}
        # Move the mouse and click within the window
        if action["click"]:
            subprocess.run(["xdotool", "mousemove", "--window", self.window_id, str(action["x"]), str(action["y"])])
            subprocess.run(["xdotool", "click", "--window", self.window_id, "1"])
        # wait before next action
        time.sleep(2)

    def double_click_at_location(self, x, y):
        # Move the mouse to the specified location
        subprocess.run(["xdotool", "mousemove", "--window", self.window_id, str(int(x)), str(int(y))])
        # Double click
        subprocess.run(["xdotool", "click", "--repeat", "1", "--window", self.window_id, "1"])
        time.sleep(0.1)
        subprocess.run(["xdotool", "click", "--repeat", "1", "--window", self.window_id, "1"])

    def enter_text_at_location(self, text, x, y):
        # Move the mouse to the specified location
        subprocess.run(["xdotool", "mousemove", "--window", self.window_id, str(int(x)), str(int(y))])
        # Click to focus at the location
        subprocess.run(["xdotool", "click", "--window", self.window_id, "1"])
        # Type the text
        subprocess.run(["xdotool", "type", "--window", self.window_id, text])

    def press_enter(self):
        subprocess.run(["xdotool", "key", "--window", self.window_id, "Return"])

    def take_screenshot(self):
        # Take a screenshot
        screenshot_command = ["import", "-window", self.window_id, "screenshot.png"]
        subprocess.run(screenshot_command)
        # Wait before next action
        time.sleep(1)
        self.image = Image.open("screenshot.png").convert("RGB")
        self.aui_annotate()
        return "screenshot taken with UI elements numbered at screenshot_annotated.png "

    def aui_annotate(self):
        assert os.path.exists("screenshot.png"), "Screenshot not taken"
        self.raw_data = (request_image_annotation("screenshot.png", workspaceId, token)).json()
        self.image_with_bboxes = draw_bboxes(self.image, self.raw_data)
        self.image_with_bboxes.save("screenshot_annotated.png")
    
    def extract_location_from_index(self, index):
        bbox = extract_element_bbox([index], self.raw_data)
        return [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    
    def convert_image_to_base64(self):
        return encode_image("screenshot_annotated.png")
    
    def get_target_UIelement_number(self, query):
        base64_image = self.convert_image_to_base64()
        gpt_response = request_ui_element_gpt4v(base64_image, query)
        gpt_response = gpt_response["choices"][0]["message"]["content"]
        # Extract numbers from the GPT response
        numbers = extract_numbers(gpt_response)
        return numbers[0]

    def analyze_ui_state(self, query):
        base64_image = self.convert_image_to_base64()
        gpt_response = analyze_ui_state_gpt4v(base64_image, query)
        gpt_response = gpt_response["choices"][0]["message"]["content"]
        # Extract numbers from the GPT response
        numbers = extract_numbers(gpt_response)
        return numbers[0]
```

# 5. GPT-4 Text Agent: The Orchestrator

Finally, the Orchestrator, a GPT-4 text model, plans and executes tasks. It leverages the capabilities of the Device Controller and GPT-4v, orchestrating actions to achieve the set goal. The Orchestrator operates through a UserProxyAgent, managing tasks and monitoring progress.

## Key Components
* Planner: Breaks down goals and strategizes actions.

* UserProxyAgent: Facilitates communication between the planner and the controller.

* Controller: Executes actions within the UI.


## Workflow Example

1. Initialization: The Orchestrator receives a task.

2. Planning: It outlines the necessary steps.

3. Execution: The Controller interacts with the UI, guided by the Orchestrator’s strategy.

4. Verification: The Orchestrator checks the outcomes and adjusts actions if needed.

We use [AutoGen](https://github.com/microsoft/autogen), an open source library that lets GPTs talk to each other, to implement the Automated UI controller. After giving each GPT its system message (or identity), we register the functions so that the GPT is aware of them and can call them when needed.

```python
planner = autogen.AssistantAgent(
    name="Planner",
    system_message=
    """
     You are the orchestrator that must achieve the given task. 
     You are given functions to handle the UI window.
     Remember that you are given a UI window and you start the task by
     taking a screenshot and take screenshot after each action.
     For coding tasks, only use the functions you have been provided with.
     Reply TERMINATE when the task is done.
     Take a deep breath and think step-by-step
    """,
    llm_config=llm_config,
)


# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "coding"},
    llm_config=llm_config,
)


controller = Controller(window_name = "Mozilla Firefox")

# register the functions
user_proxy.register_function(
    function_map={
        "take_screenshot": controller.take_screenshot,
        "move_mouse": controller.move_mouse,
        "extract_location_from_index": controller.extract_location_from_index,
        "convert_image_to_base64": controller.convert_image_to_base64,
        "get_target_UIelement_number": controller.get_target_UIelement_number,
        "enter_text_at_location": controller.enter_text_at_location,
        "press_enter": controller.press_enter,
        "double_click_at_location": controller.double_click_at_location,
        "analyze_UI_image_state": controller.analyze_ui_state,
    }
)

def start_agents(query):
    user_proxy.initiate_chat(
        planner,
        message=f"Task is to: {query}. Check if the task is acheived by looking at the window. Don't quit immediately",
    )
```

# 6. Workflow on a sample task
In the Orchestrator GPT’s system message, we instruct it to take a screenshot to know the current state of UI and plan to reach to the final goal that user request. So, at each point, it takes screenshot and checks if the UI state changed accordingly or not.

Here is a workflow example on the query “click on the github icon and click on ‘blogs’ repository”. The video is available in youtube, embedded as iframe here

<figure style="max-width: 100%; width: 100%;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/-9adrW2FKac?si=kijROSl2ASRvBDpL" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    <figcaption><b>Figure 5:</b> Workflow on a sample task </figcaption>
</figure>

Attached below is a log history, performing various actions. We can see that various functions are called and executed accordingly.

<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/vision-agents/agents-log.webp"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/vision-agents/agents-log.webp"></a>
    <figcaption><b>Figure 6:</b> Log history of the task </figcaption>
</figure>

This article shows a glimpse of how the automation landscape changes in the near future. Looking forward to amazing things that will be built on this.