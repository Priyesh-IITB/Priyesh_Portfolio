import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Priyesh's Portfolio", layout="wide")

# Title and introduction
st.title("Priyesh's Portfolio")
st.write(
    "Welcome to my professional portfolio. Here you can find details about my work experience and projects."
)

# AT&T Experience
st.header("AT&T Experience")

with st.expander("Unified Dashboards Development"):
    st.subheader("Clients and Technologies")
    st.write("- Developed dashboards for clients in Pharmaceuticals, Energy (Siemens Energy), and Election Management sectors.")
    st.write("- Technologies Used: SQL, R, Tableau")
    
    st.subheader("Data Integration via APIs")
    st.write("Integrated data from Moeco and Telit platforms into a single interface, enabling clients to access consolidated information such as asset locations, sensor data (temperature, humidity, shock), and shipment tracking details.")
    
    st.subheader("Use Cases")
    st.write("1. Pharmaceuticals: Built a dashboard tracking real-time temperature and location of drug shipments using disposable smart labels.")
    st.write("2. Energy (Siemens Energy): Created a dashboard tracking location and status of expensive shipments, including tools and repair kits.")
    
    # SQL Usage Example
    st.subheader("SQL Usage Example")
    sql_code = """
    SELECT a.asset_id, a.asset_type, l.latitude, l.longitude, s.temperature, s.humidity
    FROM assets a
    JOIN locations l ON a.asset_id = l.asset_id
    JOIN sensor_data s ON a.asset_id = s.asset_id
    WHERE a.client_id = 'PHARMA_CLIENT_001'
    AND s.timestamp >= DATEADD(hour, -24, GETDATE())
    ORDER BY s.timestamp DESC;
    """
    st.code(sql_code, language='sql')
    
    # R Usage Example
    st.subheader("R Usage Example")
    r_code = """
    library(dplyr)
    library(lubridate)
    
    # Load data from SQL query result
    asset_data <- read.csv("asset_data.csv")
    
    # Clean and transform data
    cleaned_data <- asset_data %>%
      mutate(timestamp = ymd_hms(timestamp)) %>%
      filter(!is.na(temperature) & !is.na(humidity)) %>%
      mutate(temp_fahrenheit = (temperature * 9/5) + 32) %>%
      arrange(asset_id, timestamp)
    
    # Calculate rolling averages
    rolling_avg <- cleaned_data %>%
      group_by(asset_id) %>%
      mutate(rolling_temp_avg = rollmean(temperature, k = 6, fill = NA, align = "right"))
    
    # Export cleaned data for Tableau
    write.csv(rolling_avg, "cleaned_asset_data.csv", row.names = FALSE)
    """
    st.code(r_code, language='r')

with st.expander("ML-Powered ROI Calculator"):
    st.subheader("Objective")
    st.write("Developed an ROI calculator using Python, powered by machine learning, to improve investment efficiency and secure client buy-in.")
    
    st.subheader("Implementation")
    st.write("- ML Model: Predicted cost savings and investment returns based on historical shipment data, sensor readings, and logistics performance.")
    st.write("- Platform: Deployed using Streamlit on Heroku for real-time access by stakeholders.")
    
    st.subheader("Results")
    st.write("The tool demonstrated a 40% improvement in investment efficiency by optimizing routes and reducing delays.")
    
    # Example visualization
    st.subheader("ROI Improvement Visualization")
    weeks = range(8)
    roi_improvement = [5, 12, 18, 25, 30, 35, 38, 40]
    fig, ax = plt.subplots()
    ax.plot(weeks, roi_improvement, marker='o')
    ax.set_xlabel('Weeks')
    ax.set_ylabel('ROI Improvement (%)')
    ax.set_title('ROI Improvement Over Time')
    st.pyplot(fig)

with st.expander("Business Intelligence Strategy and Data Models/ETL Processes"):
    st.subheader("Objective")
    st.write("To extract actionable insights from over 100+ TB of unstructured data across different use cases.")
    
    st.subheader("Strategy")
    st.write("- BI Tools: Implemented Tableau for visualization and SQL for querying large datasets.")
    st.write("- Data Models: Built ETL processes to clean, transform, and load data from various IoT sensors and tracking devices.")
    
    st.subheader("Use Case: Siemens Energy")
    st.write("Developed models to compare the performance of different 3PL (third-party logistics) providers, which helped in optimizing logistics operations.")
    
    # Example SQL query
    st.subheader("Example SQL Query for 3PL Comparison")
    sql_code = """
    SELECT
        lp.provider_name,
        AVG(fs.delivery_time_hours) as avg_delivery_time,
        AVG(fs.shipping_cost) as avg_shipping_cost,
        COUNT(*) as total_shipments,
        SUM(CASE WHEN fs.delivery_time_hours <= dp.expected_delivery_time THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as on_time_delivery_rate
    FROM
        fact_shipments fs
    JOIN
        dim_logistics_provider lp ON fs.logistics_provider_id = lp.provider_id
    JOIN
        dim_product dp ON fs.product_id = dp.product_id
    WHERE
        fs.date_id BETWEEN 20240101 AND 20240331
    GROUP BY
        lp.provider_name
    ORDER BY
        avg_delivery_time ASC;
    """
    st.code(sql_code, language='sql')
    
    # Simulated 3PL comparison data
    st.subheader("3PL Provider Comparison")
    data = {
        'Provider': ['3PL A', '3PL B', '3PL C'],
        'Avg Delivery Time (hours)': [48, 52, 45],
        'Avg Shipping Cost ($)': [120, 110, 130],
        'On-Time Delivery Rate (%)': [95, 92, 97]
    }
    df = pd.DataFrame(data)
    st.table(df)

with st.expander("Change Impact Analysis and Stakeholder Management"):
    st.subheader("Objective")
    st.write("To increase the adoption rates of IT initiatives by 30%.")
    
    st.subheader("Methodology")
    st.write("- Conducted thorough change impact analysis to identify potential challenges in adopting new technologies.")
    st.write("- Developed detailed stakeholder plans to ensure smooth transitions and buy-in from all parties.")
    st.write("- Implemented training sessions and documentation to support stakeholders in using the new systems effectively.")
    
    st.subheader("Outcome")
    st.write("The adoption rate of new IT initiatives increased by 30% as measured by the rate of user engagement and system usage.")
    
    # Simulated adoption rate data
    st.subheader("Adoption Rate Over Time")
    weeks = range(8)
    adoption_rate = [5, 10, 15, 20, 23, 26, 28, 30]
    fig, ax = plt.subplots()
    ax.plot(weeks, adoption_rate, marker='o')
    ax.set_xlabel('Weeks')
    ax.set_ylabel('Adoption Rate Increase (%)')
    ax.set_title('IT Initiative Adoption Rate')
    st.pyplot(fig)
    
    # Example stakeholder matrix
    st.subheader("Stakeholder Matrix")
    matrix_data = {
        'Stakeholder': ['Executive Team', 'IT Department', 'End Users', 'External Clients'],
        'Influence': [9, 8, 6, 7],
        'Interest': [7, 9, 8, 6],
        'Engagement Strategy': ['Regular meetings', 'Involve in decision-making', 'Comprehensive training', 'Keep informed']
    }
    matrix_df = pd.DataFrame(matrix_data)
    st.table(matrix_df)

# Ernst & Young Experience
st.header("Ernst & Young Experience")

with st.expander("Expansion Plan for a National Bank"):
    st.subheader("Objective")
    st.write("Led a cross-functional team to develop a comprehensive expansion plan for a leading national bank into the ASEAN markets.")
    
    st.subheader("Strategy")
    st.write("- Conducted market research and competitive analysis to identify potential entry points.")
    st.write("- Worked closely with legal teams to ensure regulatory compliance in target countries.")
    
    st.subheader("Outcome")
    st.write("The plan enabled the bank's successful entry into two ASEAN markets, positioning them for long-term growth.")
    
    # Python Code Example for Market Analysis
    python_code = """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    def analyze_asean_markets(market_data):
        # Perform market analysis
        market_potential = market_data.groupby('country')['market_size'].sum().sort_values(ascending=False)
        
        # Visualize market potential
        plt.figure(figsize=(10, 6))
        market_potential.plot(kind='bar')
        plt.title('ASEAN Market Potential')
        plt.xlabel('Country')
        plt.ylabel('Market Size (USD Millions)')
        plt.savefig('asean_market_potential.png')
        plt.close()
        
        return market_potential
    
    def regulatory_compliance_check(regulations_data):
        compliance_score = regulations_data.groupby('country')['compliance_ease'].mean()
        return compliance_score
    
    # Usage
    market_data = pd.read_csv('asean_market_data.csv')
    regulations_data = pd.read_csv('asean_regulations.csv')
    
    market_potential = analyze_asean_markets(market_data)
    compliance_score = regulatory_compliance_check(regulations_data)
    
    # Combine market potential and compliance score for decision-making
    expansion_targets = pd.concat([market_potential, compliance_score], axis=1)
    top_targets = expansion_targets.nlargest(2, 'market_size')
    print("Top expansion targets:", top_targets)
    """
    st.code(python_code, language='python')

with st.expander("Digital Transformation Project Optimization"):
    st.subheader("Objective")
    st.write("Optimized a $20M digital transformation project by assessing and redesigning the client's organizational structure.")
    
    st.subheader("Methodology")
    st.write("- Conducted detailed process mapping to identify bottlenecks and inefficiencies.")
    st.write("- Proposed a new organizational model focused on cross-functional collaboration and streamlined decision-making.")
    
    st.subheader("Outcome")
    st.write("The new structure improved operational efficiency by 15%, reducing project timelines and costs.")
    
    # Python Code Example for Process Mapping
    python_code = """
    import networkx as nx
    
    def map_processes(processes):
        G = nx.DiGraph()
        for process in processes:
            G.add_edge(process['start'], process['end'], weight=process['duration'])
        return G
    
    def identify_bottlenecks(G):
        bottlenecks = [node for node in G.nodes() if G.in_degree(node) > 2 and G.out_degree(node) > 2]
        return bottlenecks
    
    def propose_new_structure(G, bottlenecks):
        new_structure = G.copy()
        for bottleneck in bottlenecks:
            # Implement cross-functional teams at bottlenecks
            new_structure.add_node(f"Cross-functional team {bottleneck}")
            for predecessor in G.predecessors(bottleneck):
                new_structure.add_edge(predecessor, f"Cross-functional team {bottleneck}")
            for successor in G.successors(bottleneck):
                new_structure.add_edge(f"Cross-functional team {bottleneck}", successor)
            new_structure.remove_node(bottleneck)
        return new_structure
    
    # Usage
    processes = [
        {'start': 'Ideation', 'end': 'Planning', 'duration': 2},
        {'start': 'Planning', 'end': 'Development', 'duration': 5},
        # ... more processes
    ]
    
    current_structure = map_processes(processes)
    bottlenecks = identify_bottlenecks(current_structure)
    new_structure = propose_new_structure(current_structure, bottlenecks)
    
    # Visualize the new structure
    nx.draw(new_structure, with_labels=True)
    plt.savefig('new_organizational_structure.png')
    plt.close()
    """
    st.code(python_code, language='python')

with st.expander("Market Entry Strategy for VISA"):
    st.subheader("Objective")
    st.write("Spearheaded the development of a market entry strategy for VISA to enter the port sector, focusing on B2B opportunities.")
    
    st.subheader("Strategy")
    st.write("- Identified key B2B opportunities within the port sector through extensive market research.")
    st.write("- Designed a targeted approach that included strategic partnerships and customized product offerings.")
    
    st.subheader("Outcome")
    st.write("The strategy led to a 9% increase in revenue, successfully positioning VISA as a key player in the new market.")
    
    # Python Code Example for Market Segmentation
    python_code = """
    import pandas as pd
    from sklearn.cluster import KMeans
    
    def analyze_port_sector(port_data):
        # Perform K-means clustering to identify key segments
        kmeans = KMeans(n_clusters=3, random_state=42)
        port_data['segment'] = kmeans.fit_predict(port_data[['transaction_volume', 'revenue_potential']])
        
        return port_data
    
    def identify_partnership_opportunities(port_data, partner_data):
        # Merge port data with potential partner data
        opportunities = pd.merge(port_data, partner_data, on='port_id')
        
        # Score partnership opportunities
        opportunities['partnership_score'] = (
            opportunities['transaction_volume'] * 0.4 +
            opportunities['revenue_potential'] * 0.4 +
            opportunities['partner_reliability'] * 0.2
        )
        
        return opportunities.nlargest(5, 'partnership_score')
    
    # Usage
    port_data = pd.read_csv('port_sector_data.csv')
    partner_data = pd.read_csv('potential_partners.csv')
    
    segmented_ports = analyze_port_sector(port_data)
    top_opportunities = identify_partnership_opportunities(segmented_ports, partner_data)
    print("Top partnership opportunities:", top_opportunities)
    """
    st.code(python_code, language='python')

with st.expander("Go-to-Market Strategy for a New Product Launch"):
    st.subheader("Objective")
    st.write("Designed and executed a data-driven go-to-market strategy to capture significant market share for a new product.")
    
    st.subheader("Approach")
    st.write("- Leveraged data analytics to understand customer behavior and preferences.")
    st.write("- Developed and implemented targeted marketing campaigns tailored to specific customer segments.")
    
    st.subheader("Outcome")
    st.write("The strategy captured a 25% market share within six months, exceeding the client's expectations.")
    
    # Python Code Example for Customer Segmentation
    python_code = """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    def segment_customers(customer_data):
        # Preprocess data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(customer_data[['purchase_frequency', 'avg_order_value', 'customer_lifetime_value']])
        
        # Perform customer segmentation
        kmeans = KMeans(n_clusters=4, random_state=42)
        customer_data['segment'] = kmeans.fit_predict(scaled_data)
        
        return customer_data
    
    def design_targeted_campaigns(segmented_customers):
        campaigns = {}
        for segment in segmented_customers['segment'].unique():
            segment_data = segmented_customers[segmented_customers['segment'] == segment]
            if segment_data['avg_order_value'].mean() > 1000:
                campaigns[segment] = "Premium product focus, loyalty rewards"
            elif segment_data['purchase_frequency'].mean() > 10:
                campaigns[segment] = "High-frequency engagement, cross-selling"
            else:
                campaigns[segment] = "Awareness building, introductory offers"
        return campaigns
    
    # Usage
    customer_data = pd.read_csv('customer_data.csv')
    segmented_customers = segment_customers(customer_data)
    targeted_campaigns = design_targeted_campaigns(segmented_customers)
    print("Targeted campaigns:", targeted_campaigns)
    """
    st.code(python_code, language='python')

with st.expander("Pre-Sales Support and Proposal Development"):
    st.subheader("Objective")
    st.write("Supported EY's pre-sales efforts by crafting compelling proposals and providing tailored solutions to prospective clients.")
    
    st.subheader("Strategy")
    st.write("- Conducted thorough research to understand client needs and challenges.")
    st.write("- Designed customized solutions that addressed specific client pain points, enhancing the chances of winning contracts.")
    
    st.subheader("Outcome")
    st.write("Contributed to the successful acquisition of multiple high-value contracts, driving revenue growth for EY.")
    
    # Python Code Example for Client Needs Analysis
    python_code = """
    import nltk
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    nltk.download('stopwords')
    
    def analyze_client_needs(client_documents):
        # Preprocess documents
        stop_words = set(stopwords.words('english'))
        vectorizer = TfidfVectorizer(stop_words=stop_words)
        tfidf_matrix = vectorizer.fit_transform(client_documents)
        
        # Identify key themes
        feature_names = vectorizer.get_feature_names_out()
        key_themes = []
        for doc_index in range(len(client_documents)):
            top_features = tfidf_matrix[doc_index].toarray()[0].argsort()[-5:][::-1]
            key_themes.append([feature_names[i] for i in top_features])
        
        return key_themes
    
    def tailor_proposal(key_themes, solution_templates):
        tailored_proposal = ""
        for theme in key_themes:
            if theme in solution_templates:
                tailored_proposal += solution_templates[theme] + "\\n\\n"
        return tailored_proposal
    
    # Usage
    client_documents = [
        "The client is facing challenges with digital transformation and data security.",
        "There's a need for improved customer engagement and loyalty programs.",
        # ... more client documents
    ]
    
    solution_templates = {
        "digital transformation": "Our digital transformation solution includes...",
        "data security": "We offer state-of-the-art data security measures...",
        "customer engagement": "Our customer engagement strategy focuses on...",
        # ... more solution templates
    }
    
    key_themes = analyze_client_needs(client_documents)
    proposal = tailor_proposal(key_themes[0], solution_templates)
    print("Tailored proposal:", proposal)
    """
    st.code(python_code, language='python')

# Cook County Assessor's Office Experience
st.header("Cook County Assessor's Office Experience")

with st.expander("Developing Scalable AWS Data Infrastructure"):
    st.subheader("Objective")
    st.write("Built a scalable data infrastructure to optimize the valuation process for over 2 million residential units across Cook County.")
    
    st.subheader("Implementation")
    st.write("- Utilized AWS services like Amazon S3, Amazon Redshift, and AWS Glue for data storage, warehousing, and ETL processes.")
    st.write("- Ensured the infrastructure could automatically scale based on data load, enhancing performance during peak times.")
    
    st.subheader("Outcome")
    st.write("The infrastructure improved the office's ability to process and analyze property data, leading to more accurate valuations.")
    
    # R Code Example for AWS Interaction
    r_code = """
    library(aws.s3)
    library(aws.ec2)
    
    # Set up connection to S3
    Sys.setenv("AWS_ACCESS_KEY_ID" = "your_access_key",
               "AWS_SECRET_ACCESS_KEY" = "your_secret_key")
    
    # Upload data to S3
    put_object(file = "property_data.csv", 
               bucket = "cook-county-property-data",
               object = "raw_data/property_data.csv")
    
    # Read data from S3
    property_data <- s3read_using(FUN = read.csv, 
                                  bucket = "cook-county-property-data",
                                  object = "raw_data/property_data.csv")
    """
    st.code(r_code, language='r')

with st.expander("Advanced Feature Engineering Techniques"):
    st.subheader("Objective")
    st.write("Applied advanced feature engineering techniques to improve the accuracy of property valuation models by 30%.")
    
    st.subheader("Methodology")
    st.write("- Conducted data exploration to identify key features like proximity to public transportation, property size, and recent sale prices.")
    st.write("- Used techniques like normalization, binning, and polynomial feature creation to enhance model inputs.")
    
    st.subheader("Outcome")
    st.write("The enhanced feature set led to a significant improvement in the predictive models used for property valuations.")
    
    # R Code Example for Feature Engineering
    r_code = """
    library(dplyr)
    library(caret)
    
    # Load data
    property_data <- read.csv("property_data.csv")
    
    # Create new features
    property_data <- property_data %>%
      mutate(age = 2024 - year_built,
             price_per_sqft = sale_price / total_sqft)
    
    # Normalize numerical features
    preprocess_params <- preProcess(property_data[c("total_sqft", "num_bedrooms", "num_bathrooms", "age", "price_per_sqft")], 
                                    method = c("center", "scale"))
    property_data_normalized <- predict(preprocess_params, property_data)
    
    # One-hot encode categorical features
    dummy_vars <- dummyVars(~ property_type + neighborhood, data = property_data)
    property_data_encoded <- predict(dummy_vars, property_data)
    property_data_final <- cbind(property_data_normalized, property_data_encoded)
    """
    st.code(r_code, language='r')

with st.expander("Developing Advanced Data Models & Machine Learning Algorithms"):
    st.subheader("Objective")
    st.write("Developed machine learning algorithms to predict property values more accurately, reducing assessment errors by 15%.")
    
    # R Code Example for Random Forest Model
    r_code = """
    library(randomForest)
    library(caret)
    
    # Splitting the data
    set.seed(123)
    train_index <- createDataPartition(property_data_final$sale_price, p = 0.8, list = FALSE)
    train_data <- property_data_final[train_index, ]
    test_data <- property_data_final[-train_index, ]
    
    # Train Random Forest model
    rf_model <- randomForest(sale_price ~ ., data = train_data, ntree = 500, mtry = 4)
    
    # Make predictions
    rf_predictions <- predict(rf_model, newdata = test_data)
    
    # Evaluate the model
    mse <- mean((test_data$sale_price - rf_predictions)^2)
    r2 <- 1 - (sum((test_data$sale_price - rf_predictions)^2) / sum((test_data$sale_price - mean(test_data$sale_price))^2))
    
    print(paste("Mean Squared Error:", mse))
    print(paste("R-squared:", r2))
    """
    st.code(r_code, language='r')

with st.expander("Creating an Interactive Power BI Dashboard"):
    st.subheader("Objective")
    st.write("Developed a Power BI dashboard to visualize property valuation trends, making complex data easily interpretable.")
    
    st.subheader("Outcome")
    st.write("The dashboard reduced analysis time by 20%, enabling stakeholders to make data-driven decisions more efficiently.")
    
    # R Code Example for Dashboard Data Preparation
    r_code = """
    library(readr)
    library(dplyr)
    
    # Prepare data for dashboard
    dashboard_data <- property_data_final %>%
      group_by(neighborhood) %>%
      summarize(avg_price = mean(sale_price),
                avg_sqft = mean(total_sqft),
                num_properties = n())
    
    # Export data for Power BI
    write_csv(dashboard_data, "dashboard_data.csv")
    """
    st.code(r_code, language='r')

with st.expander("Implementing Data Quality Assurance Protocols"):
    st.subheader("Objective")
    st.write("Developed and implemented data quality assurance protocols to maintain the integrity of property valuation data.")
    
    st.subheader("Outcome")
    st.write("The QA protocols significantly reduced data errors, ensuring that all valuation decisions were based on high-quality data.")
    
    # R Code Example for Data Quality Assurance
    r_code = """
    library(assertr)
    
    # Define data quality rules
    quality_rules <- function(data) {
      data %>%
        verify(sale_price > 0) %>%
        verify(total_sqft > 0) %>%
        verify(num_bedrooms >= 0) %>%
        verify(num_bathrooms >= 0) %>%
        insist(within_n_sds(3), sale_price) # Flag outliers
    }
    
    # Apply rules to incoming data
    monitor_data_quality <- function(incoming_data) {
      tryCatch(
        {
          quality_rules(incoming_data)
          return(TRUE)
        },
        error = function(e) {
          print(paste("Data quality issue detected:", e))
          return(FALSE)
        }
      )
    }
    
    # Example usage
    new_data <- read.csv("new_property_data.csv")
    if(!monitor_data_quality(new_data)) {
      print("Alert: Data quality issues detected in new data!")
    }
    """
    st.code(r_code, language='r')

# Illinois Business Consulting Experience
st.header("Illinois Business Consulting Experience")

with st.expander("R-Based Transit Analysis System"):
    st.subheader("Objective")
    st.write("Developed an R-based transit analysis system to optimize routes and reduce travel times for over 500,000 daily commuters.")
    
    st.subheader("Implementation")
    st.write("- Engineered the system using machine learning algorithms to optimize over 100 routes.")
    st.write("- Aggregated data from GPS, historical transit data, and commuter feedback to drive the analysis.")
    
    st.subheader("Outcome")
    st.write("The system improved transit efficiency by 25%, significantly reducing travel times and enhancing the overall commuter experience.")
    
    # R Code Example for Data Ingestion and Analysis
    r_code = """
    library(readr)
    library(dplyr)
    library(randomForest)
    library(lpSolve)
    
    # Load and combine data
    gps_data <- read_csv("gps_data.csv")
    historical_data <- read_csv("historical_data.csv")
    commuter_feedback <- read_csv("commuter_feedback.csv")
    
    combined_data <- gps_data %>%
      left_join(historical_data, by = "route_id") %>%
      left_join(commuter_feedback, by = "route_id")
    
    # Random Forest model for travel time prediction
    rf_model <- randomForest(travel_time ~ ., data = combined_data, ntree = 500)
    
    # Route optimization using linear programming
    objective.fn <- c(route_times)
    constraint.mat <- matrix(route_coverage, nrow = num_areas, byrow = TRUE)
    constraints.dir <- rep(">=", num_areas)
    constraints.rhs <- rep(1, num_areas)
    
    optimized_routes <- lp("min", objective.fn, constraint.mat, constraints.dir, constraints.rhs)
    """
    st.code(r_code, language='r')

with st.expander("Building SQLite Pipelines with Unit Testing for Parquet Files"):
    st.subheader("Objective")
    st.write("Implemented SQLite pipelines for processing large volumes of data stored in Parquet files, improving data processing speed by 40%.")
    
    st.subheader("Implementation")
    st.write("- Designed pipelines using RSQLite and arrow packages in R, incorporating unit testing to ensure data integrity.")
    
    st.subheader("Outcome")
    st.write("The robust pipelines enabled faster and more reliable insights, boosting the overall efficiency of data processing operations.")
    
    # R Code Example for SQLite Pipeline
    r_code = """
    library(RSQLite)
    library(arrow)
    library(testthat)
    
    # Read Parquet file
    parquet_data <- read_parquet("transit_data.parquet")
    
    # Connect to SQLite database
    con <- dbConnect(RSQLite::SQLite(), "transit_database.db")
    
    # Write data to SQLite
    dbWriteTable(con, "transit_data", parquet_data)
    
    # Example query
    result <- dbGetQuery(con, "SELECT * FROM transit_data WHERE route_id = 'A1'")
    
    dbDisconnect(con)
    
    # Unit testing for data integrity
    test_that("Data integrity is maintained", {
      original_data <- read_parquet("transit_data.parquet")
      sqlite_data <- dbGetQuery(con, "SELECT * FROM transit_data")
      
      expect_equal(nrow(original_data), nrow(sqlite_data))
      expect_equal(ncol(original_data), ncol(sqlite_data))
      expect_equal(sum(original_data$travel_time), sum(sqlite_data$travel_time), tolerance = 0.001)
    })
    """
    st.code(r_code, language='r')

with st.expander("Enhancing Transportation Planning Accuracy"):
    st.subheader("Objective")
    st.write("Applied advanced statistical models and machine learning techniques to increase the accuracy of transportation planning by 30%.")
    
    st.subheader("Implementation")
    st.write("- Utilized time series analysis and spatial statistics to analyze and predict traffic patterns.")
    st.write("- Integrated machine learning models to dynamically adjust routes based on real-time data.")
    
    st.subheader("Outcome")
    st.write("The refined route planning methods ensured more reliable and efficient transit operations, benefiting a large urban area.")
    
    # R Code Example for Time Series Analysis
    r_code = """
    library(forecast)
    library(sp)
    
    # Time series forecasting for travel times
    time_series_model <- auto.arima(route_data$travel_time)
    forecasted_times <- forecast(time_series_model, h = 24)
    
    # Spatial analysis for traffic patterns
    coordinates(route_data) <- ~longitude + latitude
    variogram_model <- variogram(travel_time ~ 1, data = route_data)
    fit_variogram <- fit.variogram(variogram_model, model = vgm("Sph"))
    """
    st.code(r_code, language='r')

with st.expander("Developing an Interactive Tableau Dashboard"):
    st.subheader("Objective")
    st.write("Created an interactive Tableau dashboard to facilitate real-time decision-making for stakeholders.")
    
    st.subheader("Implementation")
    st.write("- Integrated predictive analytics into the dashboard, allowing stakeholders to make data-driven decisions quickly.")
    
    st.subheader("Outcome")
    st.write("The dashboard reduced decision-making time by 50%, enabling more agile and responsive management of transit services.")
    
    # Example Steps for Tableau Integration
    st.subheader("Tableau Integration Steps")
    st.write("""
    1. Connected Tableau to the R-generated datasets and model outputs.
    2. Designed interactive visualizations for key metrics like route efficiency, commuter density, and predicted travel times.
    3. Implemented filters and parameters to allow stakeholders to explore different scenarios and timeframes.
    4. Created calculated fields in Tableau to display real-time predictions based on the machine learning models.
    """)

with st.expander("Designing a Data Quality Monitoring Framework"):
    st.subheader("Objective")
    st.write("Designed a data quality monitoring framework that maintained a 98% accuracy rate in real-time data feeds, supporting high-quality decision-making.")
    
    st.subheader("Implementation")
    st.write("- Developed tools to monitor data streams in real-time, flagging anomalies and inconsistencies.")
    st.write("- Implemented accuracy checks that ensured data was reliable, significantly reducing errors in transit planning.")
    
    st.subheader("Outcome")
    st.write("The framework ensured the reliability of the data used for transit planning, resulting in more accurate and efficient operations.")
    
    # R Code Example for Data Quality Monitoring
    r_code = """
    library(assertr)
    
    # Define data quality rules
    quality_rules <- function(data) {
      data %>%
        verify(travel_time > 0) %>%
        verify(latitude >= -90 & latitude <= 90) %>%
        verify(longitude >= -180 & longitude <= 180) %>%
        insist(within_n_sds(3), travel_time) # Flag outliers
    }
    
    # Apply rules to incoming data
    monitor_data_quality <- function(incoming_data) {
      tryCatch(
        {
          quality_rules(incoming_data)
          return(TRUE)
        },
        error = function(e) {
          log_error(e)
          return(FALSE)
        }
      )
    }
    
    # Continuous monitoring
    while(TRUE) {
      new_data <- get_new_data()  # Function to fetch new data
      if(!monitor_data_quality(new_data)) {
        send_alert("Data quality issue detected!")
      }
      Sys.sleep(60)  # Check every minute
    }
    """
    st.code(r_code, language='r'). myenv/bin/activate


# Born Group Experience
st.header("Born Group Experience")

with st.expander("Comparative Analysis Across 40+ Nations"):
    st.subheader("Objective")
    st.write("Conducted comparative analysis across 40+ nations to identify key competitive advantages and market opportunities for clients.")
    
    st.subheader("Implementation")
    st.write("- Gathered data from market reports, economic indicators, and industry-specific databases.")
    st.write("- Analyzed data using statistical tools in Excel and R, focusing on metrics such as GDP growth, consumer behavior, and regulatory environments.")
    
    st.subheader("Outcome")
    st.write("The insights informed 4 comprehensive business proposals, tailored to exploit opportunities in specific markets and optimize clients’ global strategies.")
    
    # R Code Example for Comparative Analysis
    r_code = """
    library(tidyverse)
    library(ggplot2)
    
    # Load and prepare data
    country_data <- read.csv("country_data.csv")
    
    # Comparative analysis
    gdp_comparison <- country_data %>%
      group_by(country) %>%
      summarize(avg_gdp_growth = mean(gdp_growth, na.rm = TRUE),
                avg_inflation = mean(inflation, na.rm = TRUE),
                avg_unemployment = mean(unemployment, na.rm = TRUE))
    
    # Visualization
    ggplot(gdp_comparison, aes(x = avg_gdp_growth, y = avg_inflation, size = avg_unemployment, color = country)) +
      geom_point() +
      labs(title = "Country Comparison: GDP Growth vs Inflation",
           x = "Average GDP Growth",
           y = "Average Inflation Rate",
           size = "Average Unemployment Rate")
    """
    st.code(r_code, language='r')

with st.expander("Strategic Roadmaps for 20+ Clients"):
    st.subheader("Objective")
    st.write("Formulated strategic roadmaps for 20+ clients, leveraging predictive modeling and correlation analysis to forecast outcomes and identify growth opportunities.")
    
    st.subheader("Implementation")
    st.write("- Used predictive modeling techniques to anticipate market trends and forecast client success in various scenarios.")
    st.write("- Developed strategic plans that included detailed timelines, resource allocation, and risk assessments.")
    
    st.subheader("Outcome")
    st.write("These roadmaps generated 6 new business opportunities for clients by identifying untapped markets and optimizing their entry strategies.")
    
    # Python Code Example for Predictive Modeling
    python_code = """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Assume we have a DataFrame 'market_data' with features and target variable
    X = market_data[['gdp_growth', 'inflation', 'population', 'internet_penetration']]
    y = market_data['market_size']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("Mean squared error:", mean_squared_error(y_test, y_pred))
    print("R-squared score:", r2_score(y_test, y_pred))
    
    # Predict market size for new scenarios
    new_scenario = [[3.5, 2.1, 70000000, 75]]
    predicted_market_size = model.predict(new_scenario)
    print("Predicted market size:", predicted_market_size[0])
    """
    st.code(python_code, language='python')

with st.expander("Market Trends and Economic Indicators Analysis"):
    st.subheader("Objective")
    st.write("Analyzed market trends and economic indicators to provide clients with up-to-date insights for their business strategies.")
    
    st.subheader("Implementation")
    st.write("- Conducted thorough research on global and regional market trends, such as shifts in consumer preferences and emerging technologies.")
    st.write("- Monitored economic indicators like inflation rates, interest rates, and unemployment levels to assess their potential impact on client businesses.")
    
    st.subheader("Outcome")
    st.write("The analysis shaped the strategic direction of the proposals, ensuring they were grounded in the latest market realities and economic conditions.")
    
    # Python Code Example for Trend Analysis
    python_code = """
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Load economic indicator data
    economic_data = pd.read_csv('economic_indicators.csv', index_col='Date', parse_dates=True)
    
    # Perform trend analysis
    def analyze_trend(data, column):
        result = seasonal_decompose(data[column], model='multiplicative', extrapolate_trend='freq')
        
        plt.figure(figsize=(12, 8))
        plt.subplot(411)
        plt.plot(data[column], label='Observed')
        plt.legend(loc='best')
        plt.subplot(412)
        plt.plot(result.trend, label='Trend')
        plt.legend(loc='best')
        plt.subplot(413)
        plt.plot(result.seasonal, label='Seasonal')
        plt.legend(loc='best')
        plt.subplot(414)
        plt.plot(result.resid, label='Residual')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
    
    # Analyze trend for GDP growth
    analyze_trend(economic_data, 'GDP_Growth')
    """
    st.code(python_code, language='python')

with st.expander("Data-Driven Insights for Proposal Development"):
    st.subheader("Objective")
    st.write("Utilized data-driven insights to enhance the accuracy and relevance of business proposals.")
    
    st.subheader("Implementation")
    st.write("- Integrated real-time data into proposals to provide clients with a clear, data-backed understanding of potential outcomes.")
    st.write("- Tailored proposals to address specific client needs and challenges, demonstrating how recommended strategies could lead to tangible business benefits.")
    
    st.subheader("Outcome")
    st.write("The data-driven approach increased client engagement and confidence in the proposed strategies, enhancing the overall effectiveness of the proposals.")
    
    # Python Code Example for Proposal Insights
    python_code = """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    def generate_proposal_insights(client_data, market_data):
        # Merge client data with market data
        combined_data = pd.merge(client_data, market_data, on='Year')
        
        # Calculate correlation between client performance and market factors
        correlation = combined_data.corr()
        
        # Visualize correlation
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm')
        plt.title('Correlation between Client Performance and Market Factors')
        plt.show()
        
        # Identify top factors influencing client performance
        top_factors = correlation['Client_Performance'].sort_values(ascending=False)[1:6]
        
        return top_factors
    
    # Example usage
    client_data = pd.read_csv('client_performance.csv')
    market_data = pd.read_csv('market_factors.csv')
    
    insights = generate_proposal_insights(client_data, market_data)
    print("Top factors influencing client performance:")
    print(insights)
    """
    st.code(python_code, language='python')

with st.expander("Actionable Recommendations for Client Decision-Making"):
    st.subheader("Objective")
    st.write("Developed actionable recommendations based on analyses to help clients achieve their business objectives.")
    
    st.subheader("Implementation")
    st.write("- Formulated specific recommendations for each client, including strategies for market entry, product positioning, and competitive differentiation.")
    st.write("- Collaborated with clients to refine and customize these recommendations, ensuring they were practical, achievable, and aligned with strategic goals.")
    
    st.subheader("Outcome")
    st.write("The recommendations provided clients with clear, actionable steps to improve their market position and drive business growth, leading to successful outcomes in several cases.")
    
    # Python Code Example for Generating Recommendations
    python_code = """
    def generate_recommendations(insights, client_goals):
        recommendations = []
        
        for factor, correlation in insights.items():
            if correlation > 0.5:
                if factor in client_goals:
                    recommendations.append(f"Increase focus on {factor} to improve performance")
                else:
                    recommendations.append(f"Consider incorporating {factor} into business strategy")
            elif correlation < -0.5:
                recommendations.append(f"Mitigate negative impact of {factor} on performance")
        
        return recommendations
    
    # Example usage
    client_goals = ['Market_Share', 'Customer_Satisfaction']
    recommendations = generate_recommendations(insights, client_goals)
    
    print("Actionable Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    """
    st.code(python_code, language='python')


# Tata Precision Experience
st.header("Tata Precision Experience")

with st.expander("Pricing Optimization Project"):
    st.subheader("Objective")
    st.write("Led a pricing optimization project aimed at improving the profitability of high-volume raw material orders.")
    
    st.subheader("Implementation")
    st.write("- Developed a dynamic pricing model using Random Forest and Gradient Boosting algorithms to predict optimal prices.")
    st.write("- Collaborated with procurement and finance teams to gather and analyze data from over 10,000 purchase orders.")
    
    st.subheader("Outcome")
    st.write("The model achieved a 92% predictive accuracy, resulting in a 5% reduction in purchase costs and a 15% increase in profit margins.")
    
    # Python Code Example for Pricing Model
    python_code = """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Load and preprocess data
    df = pd.read_csv('purchase_orders.csv')
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['delivery_date'] = pd.to_datetime(df['delivery_date'])
    df['delivery_time'] = (df['delivery_date'] - df['order_date']).dt.days
    df['log_order_quantity'] = np.log1p(df['order_quantity'])
    df = pd.get_dummies(df, columns=['material_type', 'supplier_id'])
    
    # Prepare features and target
    features = ['log_order_quantity', 'month', 'quarter', 'delivery_time', 'quality_score'] + [col for col in df.columns if col.startswith(('material_type_', 'supplier_id_'))]
    X = df[features]
    y = df['unit_price']
    
    # Train models
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    
    # Predict and evaluate
    rf_pred = rf_model.predict(X_test)
    gb_pred = gb_model.predict(X_test)
    ensemble_pred = (rf_pred + gb_pred) / 2
    mse = mean_squared_error(y_test, ensemble_pred)
    r2 = r2_score(y_test, ensemble_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")
    """
    st.code(python_code, language='python')

with st.expander("Deployment of IoT-Based Infrastructure"):
    st.subheader("Objective")
    st.write("Deployed an IoT-based infrastructure to automate data collection and enhance pricing accuracy.")
    
    st.subheader("Implementation")
    st.write("- Designed and installed smart sensors and edge devices for real-time data monitoring and analysis.")
    st.write("- Developed scripts to automate data ingestion and preprocessing, integrating data with the existing ERP system.")
    
    st.subheader("Outcome")
    st.write("The system reduced manual data entry, saving over 1.5 man-hours daily and improving pricing accuracy by 13.2%.")
    
    # Python Code Example for Edge Device Data Processing
    python_code = """
    import paho.mqtt.client as mqtt
    import json
    import time
    import random
    
    # MQTT broker details
    broker_address = "mqtt.example.com"
    broker_port = 1883
    topic = "production/sensor_data"
    
    # Create MQTT client
    client = mqtt.Client("RaspberryPi_Edge")
    client.connect(broker_address, broker_port)
    
    while True:
        # Simulate sensor data (replace with actual sensor readings)
        temperature = random.uniform(20, 30)
        humidity = random.uniform(40, 60)
        
        # Create data payload
        payload = {
            "device_id": "RP001",
            "timestamp": int(time.time()),
            "temperature": temperature,
            "humidity": humidity
        }
        
        # Publish data to MQTT topic
        client.publish(topic, json.dumps(payload))
        print(f"Published: {payload}")
        
        time.sleep(5)  # Wait for 5 seconds before next reading
    """
    st.code(python_code, language='python')

with st.expander("Real-Time Pricing Dashboard Development"):
    st.subheader("Objective")
    st.write("Developed a real-time pricing dashboard to enable quick decision-making and improve market responsiveness.")
    
    st.subheader("Implementation")
    st.write("- Integrated data from the ERP system, IoT devices, and market analysis tools.")
    st.write("- Designed the dashboard in Tableau, featuring real-time updates, predictive analytics, and supplier performance metrics.")
    
    st.subheader("Outcome")
    st.write("The dashboard increased sales by 7% and boosted overall revenue by 8%, thanks to its actionable insights and quick response times.")
    
    # SQL Code Example for Data Integration
    sql_code = """
    SELECT 
        sd.DeviceId, 
        sd.Timestamp, 
        sd.Temperature, 
        sd.Humidity, 
        po.order_quantity, 
        po.unit_price 
    FROM 
        SensorData sd 
    JOIN 
        PurchaseOrders po ON DATE(sd.Timestamp) = DATE(po.order_date)
    WHERE 
        sd.Timestamp >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)
    """
    st.code(sql_code, language='sql')

# Skills section
st.header("Skills Demonstrated")
skills = [
    "Data Integration and API Development",
    "Business Intelligence and Data Visualization",
    "Machine Learning and Predictive Analytics",
    "SQL and Database Management",
    "ETL Process Development",
    "Stakeholder Management and Change Impact Analysis",
    "Project Management and Strategy Implementation"
]
for skill in skills:
    st.write(f"- {skill}")

# Call to Action
st.header("Let's Connect")
st.write("I'm always open to discussing new opportunities and collaborations. Feel free to reach out!")
st.write("Email: priyesh.techie@gmail.com")
st.write("LinkedIn: [Priyesh's LinkedIn Profile](https://www.linkedin.com/in/priyesh-verma-iitb/)")

# Footer
st.markdown("---")
st.write("© 2024 Priyesh. All rights reserved.")
