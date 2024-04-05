import streamlit as st
import plotly.express as px
import pandas as pd
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go
import json
import numpy as np
st.set_page_config(layout="wide", page_title="Logistics Client Map")
import os
from email.message import EmailMessage
from PIL import Image
from openai import OpenAI
import re

open_ai_key = st.secrets["db_username"]
mapbox_access_token = st.secrets["mapbox_access_token"]

# Sample data for potential logistics clients
# data = {
#     "Place": ["Munich", "Nuremberg", "Augsburg", "Regensburg", "Ingolstadt", 
#               "Würzburg", "Fürth", "Erlangen", "Bayreuth", "Bamberg", 
#               "Schweinfurt", "Ulm", "Passau", "Kempten"],
#     "Latitude": [48.135124, 49.452103, 48.370544, 49.013432, 48.7631, 
#                  49.7833, 49.4667, 49.5833, 49.9326091, 49.8855223, 
#                  50.0518413, 48.407987, 48.563647, 47.731190],
#     "Longitude": [11.581981, 11.076665, 10.897790, 12.101624, 11.425, 
#                   9.9333, 11, 11.0167, 11.4091504, 10.7288004, 
#                   10.2203836, 9.991187, 13.433076, 10.318187], 
#     "Description": ["Placeholder description for Munich", "Placeholder description for Nuremberg", 
#                     "Placeholder description for Augsburg", "Placeholder description for Regensburg", 
#                     "Placeholder description for Ingolstadt", "Placeholder description for Würzburg", 
#                     "Placeholder description for Fürth", "Placeholder description for Erlangen", 
#                     "Placeholder description for Bayreuth", "Placeholder description for Bamberg", 
#                     "Placeholder description for Schweinfurt", "Placeholder description for Ulm", 
#                     "Placeholder description for Passau", "Placeholder description for Kempten"],
# }
# Euro sign for revenue column in the DataFrame (€)

# load with relative path the data is stored in the data folder speditionen_v1_2.parquet
df = pd.read_parquet("data/speditionen_v1_2.parquet")

df.loc[df["category"]=="dealer", "place_id"] = np.arange(0, df[df["category"]=="dealer"].shape[0])
df["category"] = df["category"].map({"client": "Logistics Company", "dealer": "MB Truck Niederlassung"}).fillna("Logistics Company")
df["capped_revenue"] = df["Latest Revenue Number"].fillna(0).apply(lambda x: max(min(x, 30),5))
df.loc[df["category"]=="MB Truck Niederlassung", "capped_revenue"] = 12

df["Latest Revenue Number"] = df["Latest Revenue Number"].apply(lambda x: f"{str(x)} Million €" if x else "N/A")

df = df.reset_index(drop=True).drop_duplicates(subset="place_id")
df = df.rename(columns={"vicinity": "Address", "website": "Website"})
# Convert the data dictionary to a DataFrame
px.set_mapbox_access_token(mapbox_access_token)
color_map = {
        0: 'rgba(135, 206, 250, 0.8)',  # Example category name
        1: 'rgba(255, 99, 71, 0.8)'     # Another example category name
    }

if 'selected_index' not in st.session_state:
    st.session_state.selected_index = None

if "download_button" not in st.session_state:
    st.session_state.download_button = False

if "download_data" not in st.session_state:
    st.session_state.download_data = None

# Function to create the map with multiple markers
def create_map(df):
    # Define your color map for the categories
    color_map = {
        # 'Logistic Company (operates Mercedes)': px.colors.qualitative.Pastel1[2],
        'Logistic Company (fleet includes Mercedes trucks)':  px.colors.qualitative.Pastel1[1],
        'Logistic Company (fleet likely does not include Mercedes trucks)':  px.colors.qualitative.Pastel1[0],
        'Logistic Company (no brand information of fleet)': 'grey',  # Assuming you have a category like this
        'MB Truck Niederlassung': px.colors.qualitative.Pastel1[8]  # For dealerships
    }

    # Normalize 'capped_revenue' for size, mapping the range [4, 100] to [4, 20]
    max_size = 20
    min_size = 4
    df['normalized_size'] = df['capped_revenue'].apply(lambda x: (x - 4) / (40 - 4) * (max_size - min_size) + min_size)

    # Separating the dataframe
    df_logistics = df[df['category'] != "MB Truck Niederlassung"]
    df_dealerships = df[df['category'] == "MB Truck Niederlassung"]

    fig = go.Figure()

    # Add logistics companies as Scattermapbox traces
    for category, color in color_map.items():
        if category != 'MB Truck Niederlassung':
            df_filtered = df_logistics[df_logistics['competitor_indicator_text'] == category]
            hover_text = df_filtered.apply(lambda row: f"{row['name']}<br>Website: {row['Website']}<br>Address: {row['Address']}<br>Latest Revenue: {row['Latest Revenue Number']}", axis=1)
            
            fig.add_trace(go.Scattermapbox(
                lat=df_filtered['lat'],
                lon=df_filtered['long'],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=df_filtered['normalized_size'],
                    color=color,
                ),
                text=hover_text,
                hoverinfo='text',
                name=category
            ))

    # Add dealerships with a different symbol
    # Dealerships
    if not df_dealerships.empty:  # Check if dealership data is present
        hover_text_dealerships = df_dealerships.apply(lambda row: f"{row['name']}<br>Website: {row['Website']}<br>Address: {row['Address']}", axis=1)
        fig.add_trace(go.Scattermapbox(
            lat=df_dealerships['lat'],
            lon=df_dealerships['long'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=12,
                color=px.colors.qualitative.Dark2[7],
                symbol='square'
                # Removing symbol to use default, ensuring visibility
            ),
            text=hover_text_dealerships,
            hoverinfo='text',
            name='MB Truck Niederlassung'
        ))

    # Update layout with your mapbox access token
    fig.update_layout(
        mapbox=dict(
            accesstoken=mapbox_access_token,
            center=dict(lat=df['lat'].mean(), lon=df['long'].mean()),
            zoom=6.6,
            style='light'
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        legend=dict(
            title_text='',
            x=0.01,
            y=0.99,
            traceorder="normal",
            font=dict(family="sans-serif", size=10, color="black"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="Black",
            borderwidth=1,
            itemsizing='constant',
        ),
        showlegend=True,
    )

    fig.update_layout(clickmode='event+select',
                      uirevision='foo',
                  margin = dict(l=0, r=0, t=0, b=0)
                  )

    return fig

# App layout using columns
col1, col2 = st.columns([2, 1], gap="large")

# Using the column context for the map
with col1:
    st.header("Explore Truck Sales Opportunities in the Logistics Sector")
    st.caption("Interactive map showcasing logistics companies across Bavaria. Dive into the logistics landscape, "
               "identify prospects, and explore potential sales opportunities and learn more about the companies by clicking on a marker.")
    map_figure = create_map(df)
    # Capture click events on the map
    selected_points = plotly_events(map_figure, click_event=True, override_height=650, key="plotly_map")

    # if selected_points:
    #     st.session_state.selected_index = selected_points[0]['pointIndex']

# Using the column context for the place description
with col2:
    st.header("Company Details")
    if selected_points:
        print(selected_points)
        selected_index = selected_points[0].get('pointIndex')

        # Old
        # curveNumberDict = {0: "Logistics Company", 1: "MB Truck Niederlassung"}
        # curveNumber = selected_points[0].get('curveNumber')
        # marker_type = curveNumberDict[curveNumber]
        # selected_place = df[df["category"]==marker_type].iloc[selected_index]

        curveNumberDict = {0: 'Logistic Company (fleet includes Mercedes trucks)',
1: 'Logistic Company (fleet likely does not include Mercedes trucks)',
2: 'Logistic Company (no brand information of fleet)',
3: 'MB Truck Niederlassung'}
        curveNumber = selected_points[0].get('curveNumber')
        marker_type = curveNumberDict[curveNumber]
        selected_place = df[df["competitor_indicator_text"]==marker_type].iloc[selected_index]

        st.subheader(selected_place['name'])
        st.markdown(selected_place["short_description"])
        st.markdown("Link to client website: " + selected_place['Website'])
        st.markdown("Contact: " + f"\n - Phone: {selected_place['phone']} \n - Mail: {selected_place['mail']} \n - Address: {selected_place['Address']}")
        # st.markdown("Address: " + selected_place['vicinity'])
        with st.expander("### Detailed company description"):
            st.markdown(selected_place['company_info_processed_summary'])

        st.markdown("### Current financials:")
        st.markdown(selected_place["financials_description"])
        with st.expander("### Current financials details"):
            data = json.loads(selected_place["financials"])
            fig = go.Figure(data=[
                go.Bar(x=list(data.keys()), y=list(data.values()), marker_color='lightblue')
            ])

            # Update the layout
            fig.update_layout(
                title='Annual Revenue over the years',
                xaxis=dict(title='Year'),
                yaxis=dict(title='Revenue in Million Euros'),
                plot_bgcolor='black',  # Set the background color to white
                width=500,
            )

            fig.update_layout(
                xaxis=dict(showgrid=True, gridcolor='white'),
                yaxis=dict(showgrid=True, gridcolor='white')
            )
            st.plotly_chart(fig)
            st.markdown("Source: " + selected_place["financials_source"])

        st.markdown("### Fleet information")
        st.markdown(selected_place["fleet_content"])
        with st.expander("### Fleet information details and source"):
            # extract all images of the folder maps_exploration/customer_images/Scheyer and show them
            image_files = selected_place["fleet_images"]
            for image_file in image_files:
                st.image(image_file, use_column_width=True)
            st.markdown("Source: " + selected_place["Website"])



           


        # st.markdown("### Client Potential:")


    else:
        st.write("Select a company on the map to view their details.")


# Make sure to have OpenAI and any other necessary libraries installed and imported

client = OpenAI(api_key=open_ai_key)

# Your existing app code here up to the end of the 'with col2:' block
    
# Include custom CSS to make the message display area scrollable
# st.markdown("""
#     <style>
#         .chat-messages {
#             height: 400px; /* Adjust based on your preference */
#             overflow-y: auto;
#             border: 1px solid #e1e4e8;
#             border-radius: 5px;
#             padding: 10px;
#             margin-bottom: 20px;
#         }
#     </style>
# """, unsafe_allow_html=True)


def create_email(message, email_address):
    # Regular expression to find the subject line
    subject_regex = r"Subject: (.+?)\n"
    subject_regex_german = r"Betreff: (.+?)\n"

    # Search for the subject in the email text
    # print(message)
    match = re.search(subject_regex, message[0]["content"]) if re.search(subject_regex, message[0]["content"]) else re.search(subject_regex_german, message[0]["content"])
    subject = match.group(1) if match else "Welcome from Mercedes-Benz Trucks!"

    system_context =[ {"role": "system", "content": "Create an html email file out of the input. Do not use headings, but bold text for sections if applicable. Do not modify the content at all (only leave out the subject title)."}]
    
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=system_context + message)
    html_email = response.choices[0].message.content

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = 'becker.david@mercedes-trucks.com'
    msg['To'] = email_address

    msg.set_content('This is a fallback message for email clients that do not understand HTML')
    msg.add_alternative(html_email, subtype='html')
    email_bytes = msg.as_bytes()

    return email_bytes

icon = Image.open("data/logo.png")
user_icon = Image.open("data/user_icon.png")

st.markdown("## TrucksAI Assistant")

st.caption("I am here to help you extract more insights about the company or create outreach messages. Feel free to ask me anything!")

messages = st.container(height=700)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello, how can I help you today?"}]

for msg in st.session_state.messages:
    if msg["role"] == "assistant" or msg["role"] == "system":
        messages.chat_message(msg["role"], avatar=icon).write(msg["content"])
    else:
        messages.chat_message(msg["role"], avatar=user_icon).write(msg["content"])

import time

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)


system_context = [{"role": "system", "content": "You are an assistant helping a sales representative at Mercedes Benz truck to help make truck sales to logistics companies."}]
system_context += [{"role": "system", "content": "If there is a request to create an outreach letter, create the letter in a professional way that it is tailored to the client using the information available in the client's profile and fleet information. Write the mail in German - communication with the user stays in english though. As an additional part of the mail, invite the customer to a test drive at the closest Mercedes Benz truck dealership."}]
system_context += [{"role": "system", "content": "Closest Mercedes Benz truck dealership: Daimler Truck AG - Nutzfahrzeugzentrum Mercedes-Benz München"}]
if selected_points:
    selected_index = selected_points[0].get('pointIndex')
    curveNumberDict = {0: 'Logistic Company (fleet includes Mercedes trucks)',
1: 'Logistic Company (fleet likely does not include Mercedes trucks)',
2: 'Logistic Company (no brand information of fleet)',
3: 'MB Truck Niederlassung'}
    curveNumber = selected_points[0].get('curveNumber')
    marker_type = curveNumberDict[curveNumber]
    selected_place = df[df["competitor_indicator_text"]==marker_type].iloc[selected_index]
    system_context = system_context + [{"role": "system", "content": "Potential client for Mercedes Benz truck: \n " + selected_place['company_info_processed_summary'] + "\n\n Fleet information: " + selected_place["fleet_content"]}] #  + "\n\n Financials: " + selected_place["financials_description"]
    email_address = selected_place["mail"]

# print(system_context)
text_input = st.container()


if prompt := text_input.chat_input():
    # if not openai_api_key:
        # st.info("Please add your OpenAI API key to continue.")
        # st.stop()

    # client = OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    messages.chat_message("user", avatar=user_icon).write(prompt)
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=system_context + st.session_state.messages)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    messages.chat_message("assistant", avatar=icon).write_stream(stream_data(msg))

    if "Dear" in msg or "Sehr geehrt" in msg:
        message = [{"role": "user", "content": "Please create an HTML email for me."}]
        email_content_bytes = create_email([{"role": "assistant", "content": msg}], email_address)

        st.session_state.download_button = True
        st.session_state.download_data = email_content_bytes


if st.session_state.download_button:
    btn = messages.download_button(
            label="Download Mail",
            data=st.session_state.download_data,
            file_name="outreach_letter.eml",
            mime="message/rfc822"
        )


st.header("Discussed topics in the logistics sector")
with st.container(height=400):
    st.markdown("#### Telematics in Logistics\n- Modern telematics systems are increasingly being utilized in the trucking and logistics industry, as demonstrated by companies like Zollner Forwarding. \n- These systems enable real-time tracking of trucks, monitoring of driver's driving and rest times, and efficient communication between drivers and dispatch departments.\n- They also support paperless operations and enhance customer service through quick response to customer requests.\n\n#### Hydrogen-Powered Trucks\n- Niedermaier Spedition is leading the sustainability drive with the launch of their first hydrogen-powered truck.\n- This initiative is seen as an important step towards environmental protection and a greener future for the logistics industry.\n\n#### Mobile Storage Solutions\n- Geiger Transporte provides mobile storage solutions with their container rental service.\n- This service offers a cost-effective alternative to traditional storage spaces and includes transportation services for the container.\n\n#### Customs Clearance Services\n- Emons Spedition & Logistik offers comprehensive customs clearance services for transport orders, ensuring the smooth flow of goods across borders.\n- The company also provides customs consulting, process optimization, and training, as well as outsourcing services for import and export processing.\n\n#### Environmental Responsibility in Logistics\n- Louis Baufeld demonstrates a commitment to environmental responsibility in logistics by reducing CO2 emissions, implementing eco-friendly practices, and utilizing renewable energy sources.\n  \n#### General Logistics Services\n- Spedition Nuber GmbH and Karl Gross provide a wide range of logistics services including transport, warehouse logistics, and consulting.\n- These companies emphasize reliability, service orientation, and partnership with their customers.\n\n#### Industry News and Updates\n- Emons Spedition & Logistik provides industry news and updates, including information on upcoming exhibitions, company news, current state of the transport market, and tips for shipping and collecting goods.\n\n#### Charity Involvement\n- Geiger Transporte supports donation campaigns, showcasing social responsibility in the logistics sector.\n  \n#### Quality and Environmental Responsibility\n- Spedition Nuber GmbH values quality and environmental responsibility, indicating a trend in the industry towards sustainable practices.\n\n#### Project Logistics\n- Karl Gross specializes in project logistics, offering comprehensive logistics solutions for machines, equipment, and industrial facilities.\n")



