def get_triplet() -> str:
    prompt = """<image>
    
# Role: Flowchart Connectivity Analyst

## Profile
- **Task:** Analyze a flowchart image where a specific **arrowhead** is highlighted by a **Blue Bounding Box**.
- **Goal:** Accurately determine the arrow's direction and trace its path to identify the connected **Source Node(s)**, **End Node(s)**, and any **Condition Label(s)**.

## Analysis Steps (Rules)
1. **Locate Focus:**
   - Strictly focus on the arrowhead enclosed within the **Blue Bounding Box**.

2. **Determine Direction:**
   - Observe the orientation of the arrow inside the blue box to determine its flow direction.

3. **Trace Backwards (Find Source):**
   - Visually trace the line **backwards** to find the Originating Shape(s).
   - **Extract Text:** Read the text inside.

4. **Check for Condition Label(s) - Strict Alignment:**
   - Look for text labels along the arrow line (e.g., "Yes", "No").
   - **One-to-One Mapping:** You must identify the condition status for **EVERY** path identified in Step 3.
   - **Placeholder Rule:** If a specific path does NOT have a condition label, you must explicitly record it as **"None"**.
   - *Example:* If there are two paths, one with "Yes" and one without a label, output: `Yes || None`.

5. **Trace Forwards (Find End):**
   - Visually trace the arrow **forwards** to find the Destination Shape(s).
   - **Extract Text:** Read the text inside.

6. **Content Handling:**
   - **Transcription:** Keep original spelling/formatting.
   - **Separator:** Use `' || '` to separate contents for multiple paths.

## Output Format
- Always provide all three lines.
- Ensure the number of items separated by `||` is consistent across all three lines to maintain alignment.

    source_node: Text from Source A || Text from Source B
    target_node: Text from End A || Text from End B
    condition: Condition for A || Condition for B
"""
    return prompt.strip()


def get_bpmn_triplet() -> str:
    prompt = """<image>
# Role: BPMN Connectivity Analyst

## Profile
- **Task:** Analyze a BPMN diagram image where a specific **arrowhead** is highlighted by a **Blue Bounding Box**.
- **Goal:** Identify the arrow's source node, its containing pool/lane, any condition label, the target node, and its containing pool/lane.

## Analysis Steps
1. **Locate Focus:** Strictly focus on the arrowhead enclosed within the **Blue Bounding Box**.
2. **Determine Direction:** Observe the arrow orientation to determine flow direction.
3. **Trace Backwards (Find Source):** Follow the arrow line backwards to find the originating shape. Read the text inside. Also identify which pool or lane (the most immediate container) it belongs to.
4. **Check for Condition Label:** Look for a text label along the arrow line (e.g., "Yes", "No", "approved"). If none exists, record **"None"**.
5. **Trace Forwards (Find Target):** Follow the arrow forward to find the destination shape. Read the text inside. Also identify which pool or lane it belongs to.

## Output Format
Always provide exactly five lines in this order:

    source_node: <text inside the source shape>
    source_location: <pool or lane name that directly contains the source, or None>
    condition: <label on the arrow, or None>
    target_node: <text inside the target shape>
    target_location: <pool or lane name that directly contains the target, or None>
"""
    return prompt.strip()


def get_flowgen_triplet() -> str:
    prompt = """<image>
# Role: Flowchart Connectivity Analyst

## Profile
- **Task:** Analyze a flowchart image where a specific **arrowhead** is highlighted by a **Blue Bounding Box**.
- **Goal:** Identify the arrow's source node, its containing group (if any), any condition/edge label, the target node, and its containing group (if any).

## Analysis Steps
1. **Locate Focus:** Strictly focus on the arrowhead enclosed within the **Blue Bounding Box**.
2. **Determine Direction:** Observe the arrow orientation to determine flow direction.
3. **Trace Backwards (Find Source):** Follow the arrow line backwards to find the originating shape. Read the text inside. Also identify which group or container (the most immediate bounding box with a label) it belongs to, if any.
4. **Check for Condition Label:** Look for a text label along the arrow line (e.g., "Yes", "No", "approved"). If none exists, record **"None"**.
5. **Trace Forwards (Find Target):** Follow the arrow forward to find the destination shape. Read the text inside. Also identify which group or container it belongs to, if any.
6. If the blue bounding box covers multiple arrowheads, use " || " to separate multiple values in each field.

## Output Format
Always provide exactly five lines in this order:

    source_node: <text inside the source shape>
    source_location: <group or container name that directly contains the source, or None>
    condition: <label on the arrow, or None>
    target_node: <text inside the target shape>
    target_location: <group or container name that directly contains the target, or None>
"""
    return prompt.strip()


def get_e2e_prompt(with_image_tag: bool = True) -> str:
    """End-to-end (whole-image) prompt: describe the whole flowchart as triplets.

    Used by the v2 / E2E inference scripts. Set with_image_tag=False for models
    (e.g. MiniCPM) that inject the image separately and don't want a literal
    "<image>" token in the text prompt.
    """
    tag = "<image>" if with_image_tag else ""
    prompt = tag + """Please describe all the information in the flowchart image in the form of triplets: For each arrow labeled with X directed from node A to node B, output a triplet: <A, X, B>; For each unlabeled arrow directed from node A to node B, output a triplet: <A, connectedTo, B>; For each node A fully inside node group B, output a triplet <A, partOf, B>.

The following triplets are example outputs:
<Food Packaging Improvement, secures, Quantum Computing Integration>
<Graphene Production, catalyzes, Nanosensors>
<Graphene Production, partOf, Drug Delivery Systems>
<Nanocomposites, connectedTo, Spectroscopy>
<Nanodevice Fabrication, connectedTo, Graphene Production>
<Nanosensors, connectedTo, Nanocomposites>
<Spectroscopy, educates, Nanodevice Fabrication>"""
    return prompt
