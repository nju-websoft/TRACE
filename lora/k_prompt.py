def get_triple() -> str:
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

    source_nodes: Text from Source A || Text from Source B
    end_nodes: Text from End A || Text from End B
    condition: Condition for A || None
"""
    return prompt.strip()


def get_bpmn_triple() -> str:
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


def get_k_triple(n: int) -> str:
    """K-arrow prompt for CBD/FCB/FCA: N arrows labeled 1..N inside blue boxes, output N blocks."""
    prompt = """<image>

# Role: Flowchart Connectivity Analyst (multi-arrow)

## Profile
- **Task:** Analyze a flowchart image where {n} specific **arrowheads** are highlighted by **Blue Bounding Boxes**, each labeled with an index number (1, 2, ..., {n}).
- **Goal:** For EACH of the {n} highlighted arrows, determine its direction and trace its path to identify the connected Source Node(s), End Node(s), and Condition Label(s).

## Analysis Steps (apply to every arrow i = 1..{n})
1. Locate the arrowhead inside the **Blue Bounding Box** labeled with the index `i`.
2. Determine the arrow's flow direction from the orientation inside the box.
3. Trace backwards to find the originating shape(s) and read the text.
4. Check for condition label(s) along the arrow line. If a path lacks a label, record `None`. One-to-one alignment with source paths.
5. Trace forwards to find the destination shape(s) and read the text.
6. Keep original spelling/formatting. Use ` || ` to separate multiple paths for the SAME arrow.

## Output Format (strict)
Provide exactly {n} blocks, separated by blank lines, in numeric order. Each block starts with `arrow_<i>:` and contains exactly three lines:

    arrow_1:
    source_nodes: <text> || <text>
    end_nodes: <text> || <text>
    condition: <label> || None

    arrow_2:
    source_nodes: ...
    end_nodes: ...
    condition: ...

(continue through arrow_{n})
"""
    return prompt.format(n=n).strip()


def get_k_bpmn_triple(n: int) -> str:
    """K-arrow prompt for BPMN: N arrows labeled 1..N inside blue boxes, output N blocks of 5 fields each."""
    prompt = """<image>

# Role: BPMN Connectivity Analyst (multi-arrow)

## Profile
- **Task:** Analyze a BPMN diagram image where {n} specific **arrowheads** are highlighted by **Blue Bounding Boxes**, each labeled with an index number (1, 2, ..., {n}).
- **Goal:** For EACH of the {n} highlighted arrows, identify the source node, its pool/lane, any condition label, the target node, and its pool/lane.

## Analysis Steps (apply to every arrow i = 1..{n})
1. Locate the arrowhead inside the **Blue Bounding Box** labeled with the index `i`.
2. Determine flow direction from the arrow orientation.
3. Trace backwards to find the originating shape, read text, identify pool/lane.
4. Check for condition label on the arrow line. If none, record `None`.
5. Trace forwards to find the destination shape, read text, identify pool/lane.

## Output Format (strict)
Provide exactly {n} blocks, separated by blank lines, in numeric order. Each block starts with `arrow_<i>:` and contains exactly five lines:

    arrow_1:
    source_node: <text inside the source shape>
    source_location: <pool or lane name, or None>
    condition: <label on the arrow, or None>
    target_node: <text inside the target shape>
    target_location: <pool or lane name, or None>

    arrow_2:
    source_node: ...
    source_location: ...
    condition: ...
    target_node: ...
    target_location: ...

(continue through arrow_{n})
"""
    return prompt.format(n=n).strip()


def get_k_flowgen_triple(n: int, few_shot: int = 0) -> str:
    """K-arrow prompt for FlowGen, optionally with three format demonstrations."""
    few_shot_section = ""
    if few_shot == 3:
        demo_rows = [
            [
                ("Receive request", "Customer Service", "None", "Review request", "Customer Service"),
                ("Review request", "Customer Service", "approved", "Create order", "Order Processing"),
                ("Create order", "Order Processing", "None", "Archive order", "Order Processing"),
            ],
            [
                ("Start", "None", "valid", "Validate input", "Validation"),
                ("Validate input", "Validation", "No", "Reject input", "Validation"),
                ("Validate input", "Validation", "Yes", "Continue", "None"),
            ],
            [
                ("Prepare report", "Reporting", "submit", "Manager review", "Approval"),
                ("Manager review", "Approval", "None", "Publish report", "Reporting"),
                ("Publish report", "Reporting", "complete", "End", "None"),
            ],
        ]
        examples = []
        for example_idx, rows in enumerate(demo_rows, 1):
            blocks = []
            for arrow_idx in range(n):
                source, source_loc, condition, target, target_loc = rows[arrow_idx % len(rows)]
                blocks.append(
                    f"arrow_{arrow_idx + 1}:\n"
                    f"source_node: {source}\n"
                    f"source_location: {source_loc}\n"
                    f"condition: {condition}\n"
                    f"target_node: {target}\n"
                    f"target_location: {target_loc}"
                )
            examples.append(f"### Example {example_idx}\n" + "\n\n".join(blocks))
        few_shot_section = """

## Three formatting demonstrations
The following examples are fictional and demonstrate only the required output structure.
Never copy their node names or labels into the answer for the actual image.
Do not omit any highlighted arrow. Even when an arrow is difficult, return one complete five-field block for it.

{examples}
""".format(examples="\n\n".join(examples))
    prompt = """<image>

# Role: Flowchart Connectivity Analyst (multi-arrow)

## Profile
- **Task:** Analyze a flowchart image where {n} specific **arrowheads** are highlighted by **Blue Bounding Boxes**, each labeled with an index number (1, 2, ..., {n}).
- **Goal:** For EACH of the {n} highlighted arrows, identify the source node, its containing group (if any), any condition/edge label, the target node, and its containing group (if any).

## Analysis Steps (apply to every arrow i = 1..{n})
1. Locate the arrowhead inside the **Blue Bounding Box** labeled with the index `i`.
2. Determine flow direction from the arrow orientation.
3. Trace backwards to find the originating shape, read text, identify group/container.
4. Check for condition label on the arrow line. If none, record `None`.
5. Trace forwards to find the destination shape, read text, identify group/container.

{few_shot_section}

## Output Format (strict)
Provide exactly {n} blocks, separated by blank lines, in numeric order. Each block starts with `arrow_<i>:` and contains exactly five lines:

    arrow_1:
    source_node: <text inside the source shape>
    source_location: <group or container name, or None>
    condition: <label on the arrow, or None>
    target_node: <text inside the target shape>
    target_location: <group or container name, or None>

    arrow_2:
    source_node: ...
    source_location: ...
    condition: ...
    target_node: ...
    target_location: ...

(continue through arrow_{n})
"""
    return prompt.format(n=n, few_shot_section=few_shot_section).strip()


def get_flowgen_triple() -> str:
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
