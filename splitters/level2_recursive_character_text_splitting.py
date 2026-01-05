# Level 2: Recursive Character Text Splitting
#
# The problem with Level #1 is that we don't take into account the structure of our document at all. We simply split by
# a fix number of characters.
#
# The Recursive Character Text Splitter helps with this. With it, we'll specify a series of separators which will be
# used to split our docs.
#
# You can see the default separators for LangChain here. Let's take a look at them one by one.
#
# "\n\n" - Double new line, or most commonly paragraph breaks
# "\n" - New lines
# " " - Spaces
# "" - Characters
# I'm not sure why a period (".") isn't included on the list, perhaps it is not universal enough? If you know, let me know.
#
# This is the swiss army knife of splitters and my first choice when mocking up a quick application. If you don't know
# which splitter to start with, this is a good first bet.
#
# Once paragraphs are split, then it looks at the chunk size, if a chunk is too big, then it'll split by the next separator.
# If the chunk is still too big, then it'll move onto the next one and so forth.

from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
One of the most important things I didn't understand about the world when I was a child is the degree to which the returns for performance are superlinear.

Teachers and coaches implicitly told us the returns were linear. "You get out," I heard a thousand times, "what you put in." They meant well, but this is rarely true. If your product is only half as good as your competitor's, you don't get half as many customers. You get no customers, and you go out of business.

It's obviously true that the returns for performance are superlinear in business. Some think this is a flaw of capitalism, and that if we changed the rules it would stop being true. But superlinear returns for performance are a feature of the world, not an artifact of rules we've invented. We see the same pattern in fame, power, military victories, knowledge, and even benefit to humanity. In all of these, the rich get richer. [1]
"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 65, chunk_overlap=0)
documents = text_splitter.create_documents([text])

print('\n'.join(doc.page_content for doc in documents))