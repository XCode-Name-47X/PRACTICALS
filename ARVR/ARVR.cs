using UnityEngine;

public class RotateObject : MonoBehaviour
{
    // Speed of rotation (degrees per second)
    public float rotationSpeed = 50f;

    void Update()
    {
        // Rotate the object around the Y-axis every frame
        transform.Rotate(0, rotationSpeed * Time.deltaTime, 0);

        // Increase the object's scale over time
        transform.localScale += new Vector3(0.9f, 0.9f, 0.9f) * Time.deltaTime;
    }
}