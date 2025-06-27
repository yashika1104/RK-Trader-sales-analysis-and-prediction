CREATE DATABASE R_KTRADERS;
USE R_KTRADERS;

-- View whole data
SELECT * FROM dbo.cleaned_data;

-- Verfying all column names
SELECT COLUMN_NAME
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'cleaned_data' AND TABLE_SCHEMA = 'dbo';


-- Section 1: Data Validation & Exploration
-- 1. View sample data
SELECT TOP 10 * FROM dbo.cleaned_data;

-- 2. Check for missing values
SELECT 
    SUM(CASE WHEN [Total_Quantity_Sold] IS NULL THEN 1 ELSE 0 END) AS Missing_Quantity,
    SUM(CASE WHEN [Profit] IS NULL THEN 1 ELSE 0 END) AS Missing_Profit,
    SUM(CASE WHEN [Transaction_Date] IS NULL THEN 1 ELSE 0 END) AS Missing_TransactionDate
FROM dbo.cleaned_data;

-- 3. Data type and distribution check
SELECT 
    COUNT(*) AS TotalRecords,
    COUNT(DISTINCT [Product_Name]) AS UniqueProducts,
    COUNT(DISTINCT [Customer_Name]) AS UniqueCustomers
FROM dbo.cleaned_data;


 --Section 2: Sales Performance Analysis
 -- 4. Total sales and profit
SELECT 
    SUM([Total_Price_With_GST]) AS TotalSales,
    SUM(Profit) AS TotalProfit
FROM dbo.cleaned_data;

-- 5. Sales over time
SELECT 
    [Year], [Month], 
    SUM([Total_Price_With_GST]) AS MonthlySales
FROM dbo.cleaned_data
GROUP BY [Year], [Month]
ORDER BY [Year], [Month];

-- 6. Top 5 best-selling products
SELECT TOP 5 
    [Product_Name], SUM([Total_Quantity_Sold]) AS TotalSold
FROM dbo.cleaned_data
GROUP BY [Product_Name]
ORDER BY TotalSold DESC;

-- 7. Category-wise sales
SELECT 
    Category, 
    SUM([Total_Price_With_GST]) AS CategorySales
FROM dbo.cleaned_data
GROUP BY Category
ORDER BY CategorySales DESC;


--Section 3: Inventory & Stock Tracking
-- 8. Inventory status by product
SELECT 
    [Product_Name], 
    MAX([Inventory_Stock]) AS CurrentStock,
    SUM([Total_Quantity_Sold]) AS TotalSold
FROM dbo.cleaned_data
GROUP BY [Product_Name];

-- 9. Low stock alert
SELECT 
    [Product_Name], [Inventory_Stock]
FROM dbo.cleaned_data
WHERE [Inventory_Stock] < 10;


--Section 4: Customer Insights And Retention
-- 10. Top 5 customers by revenue
SELECT TOP 5 
    [Customer_Name], 
    SUM([Total_Price_With_GST]) AS TotalSpent
FROM dbo.cleaned_data
GROUP BY [Customer_Name]
ORDER BY TotalSpent DESC;

-- 11. Customer frequency
SELECT 
    [Customer_Name], COUNT(*) AS Transactions
FROM dbo.cleaned_data
GROUP BY [Customer_Name]
ORDER BY Transactions DESC;

--12. Customer Behavior and Retention
SELECT 
    [Customer_Name], 
    COUNT(*) AS PurchaseCount,
    SUM([Total_Price_With_GST]) AS TotalSpend
FROM dbo.cleaned_data
GROUP BY [Customer_Name]
ORDER BY PurchaseCount DESC;


-- Section 5: Pricing & Discount Impact
-- 13. Average discount given per category
SELECT 
    Category, 
    AVG([Discount]) AS AvgDiscount
FROM dbo.cleaned_data
GROUP BY Category;

-- 14. Sales impact of discount
SELECT 
    CASE 
        WHEN [Discount] BETWEEN 0 AND 10 THEN '0-10%'
        WHEN [Discount] BETWEEN 11 AND 20 THEN '11-20%'
        ELSE '21% and above'
    END AS DiscountRange,
    SUM([Total_Price_With_GST]) AS Sales
FROM dbo.cleaned_data
GROUP BY 
    CASE 
        WHEN [Discount] BETWEEN 0 AND 10 THEN '0-10%'
        WHEN [Discount] BETWEEN 11 AND 20 THEN '11-20%'
        ELSE '21% and above'
    END;


--Section 6: Profitability Analysis
-- 15. Profit by product
SELECT 
    [Product_Name], 
    SUM(Profit) AS TotalProfit
FROM dbo.cleaned_data
GROUP BY [Product_Name]
ORDER BY TotalProfit DESC;

-- 16. Most profitable categories
SELECT 
    Category, 
    SUM(Profit) AS CategoryProfit
FROM dbo.cleaned_data
GROUP BY Category
ORDER BY CategoryProfit DESC;


-- Section 7: Seasonal & Time-based Trends
-- 17. Day-of-week sales trends
SELECT 
    DATENAME(WEEKDAY, [Transaction_Date]) AS DayOfWeek,
    COUNT(*) AS Transactions,
    SUM([Total_Price_With_GST]) AS TotalSales
FROM dbo.cleaned_data
GROUP BY DATENAME(WEEKDAY, [Transaction_Date])
ORDER BY TotalSales DESC;

-- 18. Monthly trend analysis
SELECT 
    [Month], SUM([Total_Price_With_GST]) AS MonthlySales
FROM dbo.cleaned_data
GROUP BY [Month]
ORDER BY [Month];


--Section 8: Advanced KPIs & Forecast Helpers
-- 19. Average order value
SELECT 
    AVG([Total_Price_With_GST]) AS AvgOrderValue
FROM dbo.cleaned_data;

-- 20. Conversion rate estimation (dummy)
SELECT 
    COUNT(DISTINCT [Customer_Name]) * 1.0 / COUNT(*) * 100 AS ConversionRatePercent
FROM dbo.cleaned_data;

-- 21. Year-over-year growth
WITH SalesPerYear AS (
    SELECT [Year], SUM([Total_Price_With_GST]) AS TotalSales
    FROM dbo.cleaned_data
    GROUP BY [Year]
)
SELECT 
    s1.[Year],
    s1.TotalSales,
    (s1.TotalSales - s2.TotalSales) * 100.0 / NULLIF(s2.TotalSales, 0) AS YoY_Growth_Percentage
FROM SalesPerYear s1
LEFT JOIN SalesPerYear s2
ON s1.[Year] = s2.[Year] + 1;

