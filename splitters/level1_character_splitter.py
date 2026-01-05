# Level 1: Character Splitting
#
# Character splitting is the most basic form of splitting up your text. It is the process of simply dividing your text
# into N-character sized chunks regardless of their content or form.
#
# This method isn't recommended for any applications - but it's a great starting point for us to understand the basics.
#
# Pros: Easy & Simple
# Cons: Very rigid and doesn't take into account the structure of your text
# Concepts to know:
#
# Chunk Size - The number of characters you would like in your chunks. 50, 100, 100,000, etc.
# Chunk Overlap - The amount you would like your sequential chunks to overlap. This is to try to avoid cutting a single
# piece of context into multiple pieces. This will create duplicate data across chunks.

from langchain_text_splitters import CharacterTextSplitter

text = """
One of the most important things I didn't understand about the world when I was a child is the degree to which the returns for performance are superlinear.

Teachers and coaches implicitly told us the returns were linear. "You get out," I heard a thousand times, "what you put in." They meant well, but this is rarely true. If your product is only half as good as your competitor's, you don't get half as many customers. You get no customers, and you go out of business.

It's obviously true that the returns for performance are superlinear in business. Some think this is a flaw of capitalism, and that if we changed the rules it would stop being true. But superlinear returns for performance are a feature of the world, not an artifact of rules we've invented. We see the same pattern in fame, power, military victories, knowledge, and even benefit to humanity. In all of these, the rich get richer. [1]
"""

text_splitter = CharacterTextSplitter(chunk_size = 35, chunk_overlap=4, separator='')
documents = text_splitter.create_documents([text])

print('\n'.join(doc.page_content for doc in documents))

